# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""JIT loader for the fused block-Hadamard(16) + NVFP4 quantization kernel.

This compiles ``csrc/fused_hadamard_nvfp4_quant.cu`` against vLLM's device
headers (``nvfp4_utils.cuh`` / ``cuda_vec_utils.cuh``) from a vLLM source
checkout. Point ``VLLM_SRC_DIR`` at that checkout (defaults to a sibling
``../vllm`` of this repo). Requires an SM100+ GPU and CUDA >= 12.9.
"""

import functools
import os
from pathlib import Path

import torch


def _vllm_fp4_include_dir() -> str:
    override = os.environ.get("VLLM_SRC_DIR")
    candidates = []
    if override:
        candidates.append(Path(override))
    # Sibling checkout: .../users/<me>/vllm next to .../users/<me>/vllm-omni
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root.parent / "vllm")
    for root in candidates:
        fp4 = root / "csrc" / "libtorch_stable" / "quantization" / "fp4"
        if (fp4 / "nvfp4_utils.cuh").is_file():
            return str(fp4)
    raise FileNotFoundError(
        "Could not find vLLM's nvfp4_utils.cuh. Set VLLM_SRC_DIR to your vLLM "
        f"source checkout (looked in: {[str(c) for c in candidates]})."
    )


def _resolve_cuda_arch() -> str:
    """Target the live GPU's arch (e.g. 100a/103a/120a). Must build on the GPU
    node so ``get_device_capability`` reflects the real device; an 'a'
    (arch-specific) cubin does not JIT across Blackwell variants."""
    override = os.environ.get("FUSED_NVFP4_CUDA_ARCH")
    if override:
        return override
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"{major}{minor}a"
    return "100a"


@functools.lru_cache(maxsize=1)
def _load_module():
    from torch.utils.cpp_extension import load

    src = Path(__file__).resolve().parent / "csrc" / "fused_hadamard_nvfp4_quant.cu"
    arch = _resolve_cuda_arch()
    return load(
        name="vllm_omni_fused_hadamard_nvfp4",
        sources=[str(src)],
        extra_include_paths=[_vllm_fp4_include_dir()],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-DNVFP4_ENABLE_ELTS16=1",
            # nvcc (13.2) is newer than the toolkit/runtime headers (13.0) in the
            # venv's mixed nvidia/cu13 wheels; suppress CCCL's version guard.
            "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
            # Re-enable implicit half/bf16 -> float conversions that vLLM's
            # nvfp4_utils.cuh relies on; torch's JIT disables them by default.
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=bool(int(os.environ.get("FUSED_NVFP4_VERBOSE", "0"))),
    )


def normalized_hadamard16(device: torch.device) -> torch.Tensor:
    """Return the normalized 16x16 Sylvester Hadamard (H @ H.T = I), float32."""
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < 16:
        h = torch.cat(
            (torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0
        )
    return (h / 4.0).to(device=device, dtype=torch.float32).contiguous()


def rotate_hadamard16(x: torch.Tensor) -> torch.Tensor:
    """Apply the same block-16 normalized Had16 the fused kernel uses, along the
    last dim (which must be divisible by 16).

    Use this OFFLINE to rotate a linear's weights before NVFP4 quantization:
    quantize ``rotate_hadamard16(W)`` instead of ``W``. Because H is orthonormal
    and block-diagonal, ``Q(W H) @ Q(X H)^T`` reconstructs ``X W^T`` while the
    quantization sees the better-conditioned rotated values. Without this the
    fused (or QuTLASS) activation rotation is speed-only and numerically invalid.
    """
    if x.shape[-1] % 16 != 0:
        raise ValueError(f"last dim {x.shape[-1]} must be divisible by 16")
    h = normalized_hadamard16(x.device).to(x.dtype)
    return (x.reshape(-1, 16) @ h).reshape(x.shape)


# Register the JIT pybind call as a torch.library custom op with a fake
# (meta) implementation. Without this, torch.compile / Dynamo cannot trace
# _load_module() or the raw pybind call -> it graph-breaks on every linear,
# trips the recompilation limit, and falls back to eager (~35 ms/step in the
# Cosmos3 denoiser). As an opaque op with a fake impl, Dynamo keeps the whole
# denoise forward in one graph.
@torch.library.custom_op("vllm_omni::fused_hadamard_nvfp4_quant", mutates_args=())
def _fused_hadamard_nvfp4_quant_op(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    signs: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_module().fused_hadamard_nvfp4_quant(x, global_scale, signs)


@_fused_hadamard_nvfp4_quant_op.register_fake
def _fused_hadamard_nvfp4_quant_fake(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    signs: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del global_scale, signs
    m = x.shape[0]
    k = x.shape[-1]
    packed = x.new_empty((m, k // 2), dtype=torch.uint8)
    rounded_m = ((m + 127) // 128) * 128
    scale_n = k // 16
    rounded_n = ((scale_n + 3) // 4) * 4
    scales = x.new_empty((rounded_m, rounded_n), dtype=torch.float8_e4m3fn)
    return packed, scales


def fused_hadamard_nvfp4_quant(
    x: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    signs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused block-16 Hadamard (FWHT) + NVFP4 quant.

    Applies the normalized 16-point Hadamard (``normalized_hadamard16``) to each
    NVFP4 group before quantizing, fused into vLLM's elementwise quant kernel.
    Dispatches through the ``vllm_omni::fused_hadamard_nvfp4_quant`` custom op so
    it stays a single opaque node under ``torch.compile``.

    Args:
        x: [m, n] bf16/fp16, contiguous, n % 16 == 0.
        global_scale: optional scalar float32 tensor (next GEMM alpha component).
        signs: optional [16] float32 (+/-1) for a randomized Hadamard; the
            checkpoint weights must be rotated with the same signs.

    Returns:
        (packed_fp4 [m, n//2] uint8, swizzled_scales fp8_e4m3fn) in the exact
        layout of vLLM ``scaled_fp4_quant(is_sf_swizzled_layout=True)`` -- a
        drop-in for ``cutlass_scaled_fp4_mm``.
    """
    if global_scale is None:
        global_scale = torch.ones(1, device=x.device, dtype=torch.float32)
    return torch.ops.vllm_omni.fused_hadamard_nvfp4_quant(x, global_scale, signs)
