# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Wan VAE full/tiled decode vs spatial SP decode.

Example:
    torchrun --nproc-per-node 2 benchmarks/diffusion/benchmark_wan_vae_sp_height.py \
        --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --subfolder vae \
        --vae-parallel-size 2 --sample-height 720 --sample-width 1280 \
        --latent-frames 21 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)


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
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _init_distributed(args: argparse.Namespace) -> None:
    if dist.is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
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


@contextmanager
def _decode_mode(mode: str):
    old_mode = os.environ.get("VLLM_OMNI_WAN_VAE_PARALLEL_MODE")
    old_split_dim = os.environ.get("VLLM_OMNI_WAN_VAE_SPLIT_DIM")
    if mode == "sp_height":
        os.environ["VLLM_OMNI_WAN_VAE_PARALLEL_MODE"] = "sp_height"
        os.environ["VLLM_OMNI_WAN_VAE_SPLIT_DIM"] = "height"
    elif mode == "sp_width":
        os.environ["VLLM_OMNI_WAN_VAE_PARALLEL_MODE"] = "sp_width"
        os.environ["VLLM_OMNI_WAN_VAE_SPLIT_DIM"] = "width"
    else:
        os.environ.pop("VLLM_OMNI_WAN_VAE_PARALLEL_MODE", None)
        os.environ.pop("VLLM_OMNI_WAN_VAE_SPLIT_DIM", None)
    try:
        yield
    finally:
        if old_mode is None:
            os.environ.pop("VLLM_OMNI_WAN_VAE_PARALLEL_MODE", None)
        else:
            os.environ["VLLM_OMNI_WAN_VAE_PARALLEL_MODE"] = old_mode
        if old_split_dim is None:
            os.environ.pop("VLLM_OMNI_WAN_VAE_SPLIT_DIM", None)
        else:
            os.environ["VLLM_OMNI_WAN_VAE_SPLIT_DIM"] = old_split_dim


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()


def _run_decode(
    vae: DistributedAutoencoderKLWan,
    latents: torch.Tensor,
    *,
    mode: str,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> DecodeResult:
    with _decode_mode(mode):
        old_use_tiling = vae.use_tiling
        vae.use_tiling = mode != "full"
        try:
            for _ in range(warmup):
                _ = vae.decode(latents, return_dict=False)[0]
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            sample = None
            for _ in range(iterations):
                sample = vae.decode(latents, return_dict=False)[0]
            _sync(device)
            latency = (time.perf_counter() - start) / iterations
            peak_memory = None
            if device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
        finally:
            vae.use_tiling = old_use_tiling

    if dist.is_initialized() and dist.get_rank() != 0:
        sample = None
    return DecodeResult(mode=mode, latency_s=latency, peak_memory_gb=peak_memory, sample=sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model repo/path containing the Wan VAE.")
    parser.add_argument("--subfolder", default=None, help="Optional VAE subfolder, usually 'vae'.")
    parser.add_argument("--vae-parallel-size", type=int, default=1)
    parser.add_argument("--backend", default="nccl")
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
        choices=["full", "tiled", "sp_height", "sp_width"],
        default=["tiled", "sp_width"],
    )
    return parser.parse_args()


def _ordered_modes(modes: list[str]) -> list[str]:
    # SP modes patch the decoder in-place, so run reference modes first.
    order = {"full": 0, "tiled": 1, "sp_height": 2, "sp_width": 2}
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
    vae.set_parallel_size(args.vae_parallel_size)

    latents = _make_latents(args, vae, device)
    modes = _ordered_modes(args.modes)
    if "sp_height" in modes and "sp_width" in modes:
        raise ValueError("Run sp_height and sp_width in separate benchmark invocations; both patch the decoder in-place.")

    results = []
    with torch.inference_mode():
        for mode in modes:
            results.append(
                _run_decode(
                    vae,
                    latents,
                    mode=mode,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
            )

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        baseline = next((r for r in results if r.mode == "full" and r.sample is not None), None)
        if baseline is None:
            baseline = next((r for r in results if r.mode == "tiled" and r.sample is not None), None)
        for result in results:
            memory = "n/a" if result.peak_memory_gb is None else f"{result.peak_memory_gb:.3f} GB"
            print(f"{result.mode}: latency={result.latency_s:.4f}s peak_memory={memory}")
            if result.sample is not None:
                print(f"{result.mode}: shape={tuple(result.sample.shape)} dtype={result.sample.dtype}")
            if baseline is not None and result.sample is not None and result.mode != baseline.mode:
                diff = (result.sample.float() - baseline.sample.float()).abs()
                print(f"{result.mode}: max_delta={diff.max().item():.6g} mean_delta={diff.mean().item():.6g}")

    if dist.is_initialized():
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
