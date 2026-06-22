# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Wan VAE tiled and spatially-sharded (SP) decode against a non-parallel reference.

Each timed mode (``--modes``) is compared against a standard non-parallel reference decode
(untiled, single-process) which serves as the ground truth. This makes the deltas independent
of the tile-parallel path, so a tiled-decode bug cannot mask SP errors. Pass ``--skip-reference``
to disable the reference when the full decode does not fit in memory.

All ranks must see every GPU (the distributed stack maps rank -> cuda:LOCAL_RANK), so make sure
CUDA_VISIBLE_DEVICES exposes all of them and matches --nproc-per-node / --vae-parallel-size.

Works with any Wan-architecture VAE loaded via DistributedAutoencoderKLWan, including the Wan2.2
VAE used by Cosmos3 (nvidia/Cosmos3-Nano / -Super, subfolder "vae").

Example (Cosmos3 VAE):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
        benchmarks/diffusion/benchmark_wan_vae_sp_decode.py \
        --model nvidia/Cosmos3-Nano --subfolder vae \
        --vae-parallel-size 4 --sample-height 720 --sample-width 1280 \
        --latent-frames 21 --dtype bfloat16 --modes tiled sp_width

Example (Wan2.2 VAE, the same architecture Cosmos3 uses):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
        benchmarks/diffusion/benchmark_wan_vae_sp_decode.py \
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --subfolder vae \
        --vae-parallel-size 4 --sample-height 720 --sample-width 1280 \
        --latent-frames 21 --dtype bfloat16 --modes tiled sp_width

Example (Wan2.1 VAE):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
        benchmarks/diffusion/benchmark_wan_vae_sp_decode.py \
        --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --subfolder vae \
        --vae-parallel-size 4 --modes tiled sp_width
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform


@dataclass
class DecodeResult:
    mode: str
    latency_s: float
    peak_memory_gb: float | None
    sample: torch.Tensor | None


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[name]


def _device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_count = current_omni_platform.get_device_count()
    if device_count <= 0:
        return torch.device("cpu")
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is out of range: this process can only see {device_count} accelerator "
            "device(s). Each rank must be able to see all GPUs (the distributed stack maps rank -> cuda:LOCAL_RANK). "
            "This usually means CUDA_VISIBLE_DEVICES is pinned to a single device; unset it or set it to all GPUs "
            f"(e.g. CUDA_VISIBLE_DEVICES=0,1,2,3), then verify torch.accelerator.device_count() == --nproc-per-node."
        )
    device = current_omni_platform.get_torch_device(local_rank)
    current_omni_platform.set_device(device)
    return device


def _init_distributed(args: argparse.Namespace) -> None:
    if dist.is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if args.vae_parallel_size != world_size:
        raise ValueError(
            "This benchmark builds a model-parallel layout that spans the whole world, and the "
            "spatially-sharded decode requires vae_parallel_size to match the DiT group size. "
            f"Got --vae-parallel-size={args.vae_parallel_size} but WORLD_SIZE={world_size}. "
            f"Launch with --nproc-per-node {args.vae_parallel_size} (or set "
            f"--vae-parallel-size {world_size})."
        )
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend=args.backend,
    )
    initialize_model_parallel(
        sequence_parallel_size=args.vae_parallel_size,
        ulysses_degree=args.vae_parallel_size,
        backend=args.backend,
    )


def _make_latents(args: argparse.Namespace, vae: DistributedAutoencoderKLWan, device: torch.device) -> torch.Tensor:
    dtype = _dtype(args.dtype)
    if args.latents_path:
        latents = torch.load(args.latents_path, map_location=device)
        if isinstance(latents, dict):
            latents = latents["latents"]
        return latents.to(device=device, dtype=dtype)

    latent_height = args.sample_height // vae.spatial_compression_ratio
    latent_width = args.sample_width // vae.spatial_compression_ratio
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    return torch.randn(
        (args.batch_size, vae.config.z_dim, args.latent_frames, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _vae_parallel_mode(mode: str) -> str:
    """Map a benchmark mode to the corresponding DiffusionParallelConfig.vae_parallel_mode."""
    if mode in ("sp_height", "sp_width"):
        return mode
    return "tile"


def _sync(device: torch.device) -> None:
    if device.type != "cpu":
        current_omni_platform.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _run_decode(
    vae: DistributedAutoencoderKLWan,
    latents: torch.Tensor,
    *,
    mode: str,
    parallel_size: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> DecodeResult:
    old_use_tiling = vae.use_tiling
    vae.use_tiling = True
    vae.set_parallel_size(parallel_size, mode=_vae_parallel_mode(mode))
    try:
        for _ in range(warmup):
            _ = vae.decode(latents, return_dict=False)[0]
        _sync(device)
        if device.type != "cpu":
            current_omni_platform.reset_peak_memory_stats()
        start = time.perf_counter()
        sample = None
        for _ in range(iterations):
            sample = vae.decode(latents, return_dict=False)[0]
        _sync(device)
        latency = (time.perf_counter() - start) / iterations
        peak_memory = None
        if device.type != "cpu":
            peak_memory = current_omni_platform.max_memory_allocated() / (1024**3)
    finally:
        vae.use_tiling = old_use_tiling

    if dist.is_initialized() and dist.get_rank() != 0:
        sample = None
    return DecodeResult(mode=mode, latency_s=latency, peak_memory_gb=peak_memory, sample=sample)


def _reference_decode(
    vae: DistributedAutoencoderKLWan,
    latents: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> DecodeResult:
    """Standard non-parallel VAE decode (untiled, single-process) used as the delta baseline.

    This is the ground-truth reference: it disables tiling and VAE parallelism so neither the
    tile-parallel nor the spatially-sharded path is exercised. It must run before any SP decode,
    since SP patches the decoder in place. The result is timed like the other modes so its latency
    and peak memory can be compared directly.
    """
    old_use_tiling = vae.use_tiling
    vae.use_tiling = False
    vae.set_parallel_size(1, mode="tile")
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                _ = vae.decode(latents, return_dict=False)[0]
            _sync(device)
            if device.type != "cpu":
                current_omni_platform.reset_peak_memory_stats()
            start = time.perf_counter()
            sample = None
            for _ in range(iterations):
                sample = vae.decode(latents, return_dict=False)[0]
            _sync(device)
            latency = (time.perf_counter() - start) / iterations
            peak_memory = None
            if device.type != "cpu":
                peak_memory = current_omni_platform.max_memory_allocated() / (1024**3)
    finally:
        vae.use_tiling = old_use_tiling

    if dist.is_initialized() and dist.get_rank() != 0:
        sample = None
    return DecodeResult(mode="reference", latency_s=latency, peak_memory_gb=peak_memory, sample=sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model repo/path containing the Wan VAE.")
    parser.add_argument("--subfolder", default=None, help="Optional VAE subfolder, usually 'vae'.")
    parser.add_argument("--vae-parallel-size", type=int, default=1)
    parser.add_argument("--backend", default=current_omni_platform.dist_backend)
    parser.add_argument("--dtype", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--latent-frames", type=int, default=21)
    parser.add_argument("--sample-height", type=int, default=720)
    parser.add_argument("--sample-width", type=int, default=1280)
    parser.add_argument("--latents-path", default=None, help="Optional .pt tensor or {'latents': tensor} input.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["tiled", "sp_height", "sp_width"],
        default=["tiled", "sp_width"],
        help="Decode modes to time. The standard non-parallel decode is always run separately as the "
        "reference baseline (unless --skip-reference), so it is not listed here.",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip the standard non-parallel (untiled, single-process) reference decode used as the "
        "delta baseline. Use this when the full decode does not fit in memory; no deltas are reported.",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    return args


def _ordered_modes(modes: list[str]) -> list[str]:
    # SP modes patch the decoder in-place, so run the tile mode first.
    order = {"tiled": 0, "sp_height": 1, "sp_width": 1}
    return sorted(dict.fromkeys(modes), key=lambda mode: order[mode])


def main() -> None:
    args = parse_args()
    device = _device()
    _init_distributed(args)

    torch_dtype = _dtype(args.dtype)
    load_kwargs = {"torch_dtype": torch_dtype}
    if args.subfolder is not None:
        load_kwargs["subfolder"] = args.subfolder
    vae = DistributedAutoencoderKLWan.from_pretrained(args.model, **load_kwargs)
    vae.to(device=device, dtype=torch_dtype)
    vae.eval()
    vae.use_tiling = True

    latents = _make_latents(args, vae, device)
    modes = _ordered_modes(args.modes)
    if "sp_height" in modes and "sp_width" in modes:
        raise ValueError(
            "Run sp_height and sp_width in separate benchmark invocations; both patch the decoder in-place."
        )

    # Compute the ground-truth reference first (standard non-parallel decode), before any SP run
    # patches the decoder in place. All timed modes are compared against this baseline.
    reference = (
        None
        if args.skip_reference
        else _reference_decode(
            vae,
            latents,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
    )
    reference_sample = reference.sample if reference is not None else None

    results = []
    with torch.inference_mode():
        for mode in modes:
            results.append(
                _run_decode(
                    vae,
                    latents,
                    mode=mode,
                    parallel_size=args.vae_parallel_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
            )

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        printable = ([reference] if reference is not None else []) + results
        if reference is None:
            print("reference (full, non-parallel): skipped; deltas are not reported")
        for result in printable:
            label = "reference (full, non-parallel)" if result.mode == "reference" else result.mode
            memory = "n/a" if result.peak_memory_gb is None else f"{result.peak_memory_gb:.3f} GB"
            speedup = ""
            if reference is not None and result.mode != "reference" and result.latency_s > 0:
                speedup = f" speedup={reference.latency_s / result.latency_s:.2f}x"
            print(f"{label}: latency={result.latency_s:.4f}s peak_memory={memory}{speedup}")
            if result.sample is not None:
                print(f"{label}: shape={tuple(result.sample.shape)} dtype={result.sample.dtype}")
            if reference_sample is not None and result.sample is not None and result.mode != "reference":
                diff = (result.sample.float() - reference_sample.float()).abs()
                print(f"{label}: max_delta={diff.max().item():.6g} mean_delta={diff.mean().item():.6g}")

    if dist.is_initialized():
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
