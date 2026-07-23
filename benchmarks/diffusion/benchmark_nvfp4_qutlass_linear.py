# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark QuTLASS activation quantization on Cosmos3-Nano linear shapes.

The swept shapes are the ModelOpt-NVFP4-quantized linears of
``nvidia/Cosmos3-Nano`` (see ``COSMOS3_NANO_LINEARS``): the attention
q/k/v/out projections and the gated-MLP gate/up/down projections that appear in
both the UND and GEN decoder layers.

The question this benchmark answers is: *does the MR-GPTQ micro-rotation add
meaningful runtime overhead on top of a plain NVFP4 W4A4 linear?* The paper
(arXiv:2509.23202, Fig. 5) claims the online block-Hadamard is essentially free
because, for block sizes < 256, the fused transform+quantize kernel is
memory-bound ("any rotation can be applied at essentially the same cost").

To isolate the rotation cost we compare paths that share the *same* QuTLASS
kernels and GEMM, only toggling the transform:

    * ``identity``        -> pure NVFP4 RTN quant (no rotation), same kernels.
    * ``random_hadamard`` -> MR-GPTQ-style online micro-rotation.

The headline numbers are therefore intra-QuTLASS ratios:

    * ``rotation_overhead_quant`` : hadamard vs identity in ``fusedQuantizeNv``.
    * ``rotation_overhead_full``  : hadamard vs identity end-to-end.
    * ``actual_over_ideal``       : full pipeline vs bare GEMM (paper's Fig. 5).

The comparison against vLLM's own ``cutlass_scaled_fp4_mm`` is reported
separately (``qutlass_full_over_vllm_nvfp4``) because it compares two *different*
GEMM libraries and does not isolate the rotation overhead.
"""

import argparse
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import torch
from vllm import _custom_ops as ops
from vllm._custom_ops import fusedQuantizeNv
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
    CutlassNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked

from vllm_omni.quantization.qutlass_nvfp4 import (
    QutlassNvFp4LinearKernel,
    QutlassNvFp4Options,
    get_qutlass_transform,
)


def _benchmark_ms(
    function: Callable[[], Any],
    *,
    warmups: int,
    iterations: int,
) -> float:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iterations


def _make_layer(
    *,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_size: int,
    input_size: int,
    global_scale: torch.Tensor,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_size_per_partition=output_size,
        input_global_scale_inv=global_scale.clone(),
        weight=packed_weight.clone(),
        weight_scale=weight_scale[:output_size, : input_size // 16].clone(),
        alpha=torch.ones(1, device=packed_weight.device, dtype=torch.float32),
    )


def _qutlass_kernel(
    config: NvFp4LinearLayerConfig,
    *,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_size: int,
    input_size: int,
    global_scale: torch.Tensor,
    transform: str,
    block_size: int,
    seed: int,
) -> tuple[QutlassNvFp4LinearKernel, SimpleNamespace]:
    # fused_kernel=False so these kernels exercise the genuine QuTLASS
    # fusedQuantizeNv + to_blocked + matmul_nvf4 path; the fused kernel is
    # timed separately via fused_kernel_full so the comparison stays meaningful.
    options = QutlassNvFp4Options(
        transform=transform, block_size=block_size, seed=seed, fused_kernel=False
    )
    kernel = QutlassNvFp4LinearKernel(config, options=options)
    layer = _make_layer(
        packed_weight=packed_weight,
        weight_scale=weight_scale,
        output_size=output_size,
        input_size=input_size,
        global_scale=global_scale,
    )
    kernel.process_weights_after_loading(layer)
    return kernel, layer


def _bench_shape(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    block_size: int,
    seed: int,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x = torch.randn(rows, in_features, device=device, dtype=dtype)
    weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
    identity16 = torch.eye(16, device=device, dtype=dtype)
    global_scale = torch.ones(1, device=device, dtype=torch.float32)
    packed_weight, weight_scale = fusedQuantizeNv(weight, identity16, global_scale)

    config = NvFp4LinearLayerConfig()

    # vLLM's own NVFP4 path (different GEMM library; reported for context only).
    cutlass_kernel = CutlassNvFp4LinearKernel(config)
    cutlass_layer = _make_layer(
        packed_weight=packed_weight,
        weight_scale=weight_scale,
        output_size=out_features,
        input_size=in_features,
        global_scale=global_scale,
    )
    cutlass_kernel.process_weights_after_loading(cutlass_layer)

    # Two QuTLASS paths sharing the same kernels/GEMM, differing only in the
    # transform, so their ratio isolates the micro-rotation overhead.
    id_kernel, id_layer = _qutlass_kernel(
        config,
        packed_weight=packed_weight,
        weight_scale=weight_scale,
        output_size=out_features,
        input_size=in_features,
        global_scale=global_scale,
        transform="identity",
        block_size=block_size,
        seed=seed,
    )
    hd_kernel, hd_layer = _qutlass_kernel(
        config,
        packed_weight=packed_weight,
        weight_scale=weight_scale,
        output_size=out_features,
        input_size=in_features,
        global_scale=global_scale,
        transform="random_hadamard",
        block_size=block_size,
        seed=seed,
    )

    id_transform = get_qutlass_transform(
        device=device,
        dtype=dtype,
        options=id_kernel.options,
    )
    hd_transform = get_qutlass_transform(
        device=device,
        dtype=dtype,
        options=hd_kernel.options,
    )

    # Pre-quantize once so the "ideal" GEMM timing excludes quant + swizzle.
    quantized_x, row_major_scale = fusedQuantizeNv(x, hd_transform, global_scale)
    blocked_scale = to_blocked(row_major_scale, backend="triton").view(
        -1, in_features // 16
    )

    def qutlass_gemm_ideal() -> torch.Tensor:
        return torch.ops._qutlass_C.matmul_nvf4_bf16_tn(
            quantized_x,
            hd_layer.weight,
            blocked_scale,
            hd_layer.weight_scale,
            hd_layer.alpha,
        )

    def cutlass_gemm_ideal() -> torch.Tensor:
        return ops.cutlass_scaled_fp4_mm(
            quantized_x,
            hd_layer.weight,
            blocked_scale,
            hd_layer.weight_scale,
            hd_layer.alpha,
            dtype,
        )

    # Lever-1 candidate: bypass QuTLASS's fusedQuantizeNv + standalone to_blocked
    # by applying the block-Hadamard as a separate rotation and then using vLLM's
    # fused quant+swizzle (scaled_fp4_quant) + tuned GEMM (cutlass_scaled_fp4_mm).
    # This removes the separate swizzle launch at the cost of one rotation pass.
    weights_pad = int(getattr(hd_layer, "weights_padding_cols", 0) or 0)
    padded_n = in_features + weights_pad * 2

    def hadacore_rotate() -> torch.Tensor:
        # Block-16 Hadamard via a [.., 16] view (matches NVFP4 group == Had16).
        return ops.hadacore_transform(x.reshape(-1, 16), inplace=False).reshape(x.shape)

    def fused_rot_vllm_full() -> torch.Tensor:
        x_rot = hadacore_rotate()
        x_fp4, x_bs = ops.scaled_fp4_quant(
            x_rot,
            global_scale,
            is_sf_swizzled_layout=True,
            backend="cutlass",
            padded_n=padded_n,
        )
        return ops.cutlass_scaled_fp4_mm(
            x_fp4,
            hd_layer.weight,
            x_bs,
            hd_layer.weight_scale,
            hd_layer.alpha,
            dtype,
        )

    def bench(fn: Callable[[], Any]) -> float:
        return _benchmark_ms(fn, warmups=warmups, iterations=iterations)

    quant_identity = bench(lambda: fusedQuantizeNv(x, id_transform, global_scale))
    quant_hadamard = bench(lambda: fusedQuantizeNv(x, hd_transform, global_scale))
    swizzle = bench(lambda: to_blocked(row_major_scale, backend="triton"))
    qutlass_gemm = bench(qutlass_gemm_ideal)
    cutlass_gemm = bench(cutlass_gemm_ideal)
    full_identity = bench(lambda: id_kernel.apply_weights(id_layer, x))
    full_hadamard = bench(lambda: hd_kernel.apply_weights(hd_layer, x))
    cutlass_full = bench(lambda: cutlass_kernel.apply_weights(cutlass_layer, x))

    # The fused-rotation path may fail if hadacore is unavailable or the scale
    # layout mismatches; record the reason instead of aborting the whole sweep.
    hadacore_rot: float | None = None
    fused_rot_full: float | None = None
    fused_rot_error: str | None = None
    if hasattr(torch.ops._C, "hadacore_transform"):
        try:
            fused_rot_vllm_full()  # correctness/layout smoke test before timing
            hadacore_rot = bench(hadacore_rotate)
            fused_rot_full = bench(fused_rot_vllm_full)
        except Exception as exc:  # noqa: BLE001 - benchmark diagnostics only
            fused_rot_error = f"{type(exc).__name__}: {exc}"
    else:
        fused_rot_error = "hadacore_transform unavailable"

    # Fused Had16+quant kernel (our custom op) + vLLM's tuned CUTLASS GEMM. This
    # is the full-linear parity candidate; only valid when the weight needs no
    # CUTLASS K-padding (so the unpadded packed activation matches the weight).
    fused_kernel_full: float | None = None
    fused_kernel_error: str | None = None
    if weights_pad:
        fused_kernel_error = f"weights_padding_cols={weights_pad} (unpadded path only)"
    else:
        try:
            from vllm_omni.quantization.fused_hadamard_nvfp4 import (
                fused_hadamard_nvfp4_quant,
            )

            def fused_kernel_full_fn() -> torch.Tensor:
                x_fp4, x_sf = fused_hadamard_nvfp4_quant(x, global_scale)
                return ops.cutlass_scaled_fp4_mm(
                    x_fp4,
                    hd_layer.weight,
                    x_sf,
                    hd_layer.weight_scale,
                    hd_layer.alpha,
                    dtype,
                )

            fused_kernel_full_fn()  # smoke test before timing
            fused_kernel_full = bench(fused_kernel_full_fn)
        except Exception as exc:  # noqa: BLE001 - benchmark diagnostics only
            fused_kernel_error = f"{type(exc).__name__}: {exc}"

    def ratio(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator <= 0:
            return None
        return numerator / denominator

    return {
        "rows": rows,
        "in_features": in_features,
        "out_features": out_features,
        "block_size": block_size,
        # Component timings (ms).
        "quant_identity_ms": quant_identity,
        "quant_hadamard_ms": quant_hadamard,
        "scale_swizzle_ms": swizzle,
        "qutlass_gemm_ms": qutlass_gemm,
        "cutlass_gemm_ms": cutlass_gemm,
        "qutlass_full_identity_ms": full_identity,
        "qutlass_full_hadamard_ms": full_hadamard,
        "vllm_nvfp4_full_ms": cutlass_full,
        "hadacore_rot_ms": hadacore_rot,
        "fused_rot_vllm_full_ms": fused_rot_full,
        "fused_rot_error": fused_rot_error,
        "fused_kernel_full_ms": fused_kernel_full,
        "fused_kernel_error": fused_kernel_error,
        # Headline overhead ratios (closer to 1.0 == cheaper rotation).
        "rotation_overhead_quant": ratio(quant_hadamard, quant_identity),
        "rotation_overhead_full": ratio(full_hadamard, full_identity),
        "actual_over_ideal": ratio(full_hadamard, qutlass_gemm),
        # Cross-library context (NOT a rotation-overhead measurement).
        "qutlass_full_over_vllm_nvfp4": ratio(full_hadamard, cutlass_full),
        "qutlass_gemm_over_vllm_gemm": ratio(qutlass_gemm, cutlass_gemm),
        # Lever-1 result: fused-rotation path vs current QuTLASS and vs vLLM.
        "fused_rot_over_qutlass_full": ratio(fused_rot_full, full_hadamard),
        "fused_rot_over_vllm_nvfp4": ratio(fused_rot_full, cutlass_full),
        # Fused Had16+quant kernel full-linear: the parity result that matters.
        "fused_kernel_over_vllm_nvfp4": ratio(fused_kernel_full, cutlass_full),
        "fused_kernel_over_qutlass_full": ratio(fused_kernel_full, full_hadamard),
    }


# Cosmos3-Nano quantized linears (ModelOpt NVFP4 wraps these). Values come from
# nvidia/Cosmos3-Nano transformer/config.json: hidden_size=4096, head_dim=128,
# num_attention_heads=32 (Q width 4096), num_key_value_heads=8 (KV width 1024),
# intermediate_size=12288. Both the UND causal-attention and GEN cross-attention
# decoder layers use this identical set. "column" shards out_features under TP;
# "row" shards in_features. proj_in/proj_out/time_embedder/action/audio
# projections are plain nn.Linear and are not quantized.
_HIDDEN = 4096
_Q_WIDTH = 32 * 128  # num_attention_heads * head_dim
_KV_WIDTH = 8 * 128  # num_key_value_heads * head_dim
_INTERMEDIATE = 12288

COSMOS3_NANO_LINEARS: dict[str, tuple[int, int, str]] = {
    # name -> (in_features, out_features, parallel)
    "to_q": (_HIDDEN, _Q_WIDTH, "column"),
    "to_kv": (_HIDDEN, _KV_WIDTH, "column"),  # to_k and to_v share this shape
    "to_out": (_Q_WIDTH, _HIDDEN, "row"),
    "gate_up": (_HIDDEN, _INTERMEDIATE, "column"),  # gate_proj and up_proj
    "down": (_INTERMEDIATE, _HIDDEN, "row"),
}

# Default M (= flattened tokens) sweep spanning Cosmos3-Nano GEN workloads:
# T2I 1024^2 (~1024 GEN tokens) up to 720p T2V 189 frames (~44k GEN tokens).
_DEFAULT_ROWS = "1024,4096,16384,44160"


def _sharded_features(in_features: int, out_features: int, parallel: str, tp: int) -> tuple[int, int]:
    if tp <= 1:
        return in_features, out_features
    if parallel == "column":
        return in_features, out_features // tp
    if parallel == "row":
        return in_features // tp, out_features
    raise ValueError(f"unknown parallel type {parallel!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=str,
        default=_DEFAULT_ROWS,
        help="Single M or comma-separated sweep of flattened token counts.",
    )
    parser.add_argument(
        "--linears",
        type=str,
        default=",".join(COSMOS3_NANO_LINEARS),
        help=f"Comma-separated subset of {list(COSMOS3_NANO_LINEARS)}.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor-parallel degree; shards column out_features / row in_features.",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    row_values = [int(value) for value in args.rows.split(",") if value.strip()]
    linear_names = [name.strip() for name in args.linears.split(",") if name.strip()]
    unknown = [name for name in linear_names if name not in COSMOS3_NANO_LINEARS]
    if unknown:
        raise ValueError(f"unknown linears {unknown}; choose from {list(COSMOS3_NANO_LINEARS)}")

    linears: list[dict[str, Any]] = []
    for name in linear_names:
        base_in, base_out, parallel = COSMOS3_NANO_LINEARS[name]
        in_features, out_features = _sharded_features(base_in, base_out, parallel, args.tp)
        if in_features % args.block_size != 0 or in_features % 16 != 0:
            raise ValueError(
                f"{name}: in_features {in_features} (tp={args.tp}) must be "
                f"divisible by block-size {args.block_size} and 16"
            )
        linears.append(
            {
                "name": name,
                "parallel": parallel,
                "in_features": in_features,
                "out_features": out_features,
                "sweep": [
                    _bench_shape(
                        rows=rows,
                        in_features=in_features,
                        out_features=out_features,
                        block_size=args.block_size,
                        seed=args.seed,
                        warmups=args.warmups,
                        iterations=args.iterations,
                    )
                    for rows in row_values
                ],
            }
        )

    results = {
        "gpu": torch.cuda.get_device_name(),
        "model": "nvidia/Cosmos3-Nano",
        "tp": args.tp,
        "block_size": args.block_size,
        "seed": args.seed,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "rows": row_values,
        "linears": linears,
    }

    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(f"{rendered}\n")


if __name__ == "__main__":
    main()
