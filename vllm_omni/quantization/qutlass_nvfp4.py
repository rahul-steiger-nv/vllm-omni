# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental block-Hadamard NVFP4 activation quantization for ModelOpt linears.

Two backends produce the online rotation + NVFP4 quantization for MR-GPTQ
(arXiv:2509.23202):

* ``fused_kernel=True`` (default, ``block_size == 16``, SM100+/CUDA>=12.9):
  ``vllm_omni.quantization.fused_hadamard_nvfp4`` -- an in-register FWHT Had16
  folded into vLLM's elementwise NVFP4 quant, emitting swizzled scale factors,
  then ``cutlass_scaled_fp4_mm``. Runs at ~pure-NVFP4 speed (1.3-4x faster than
  the QuTLASS path). Uses the fixed normalized Sylvester Had16.

* fallback: QuTLASS ``fusedQuantizeNv`` (CUTLASS GEMM) + ``to_blocked`` +
  ``matmul_nvf4_bf16_tn``; supports block sizes 16/32/64/128 and the seeded
  randomized Hadamard.

Numerical correctness note: both paths only rotate the *activations*. For a
model whose output matches the unrotated reference, the checkpoint weights must
be rotated with the SAME transform at quantization time (this is what MR-GPTQ
does offline). With a standard (unrotated) ModelOpt NVFP4 checkpoint these paths
are speed-only. The fused kernel uses the plain normalized Sylvester Had16, so a
matching MR-GPTQ checkpoint must use that same Had16 on its weights.
"""

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from vllm._custom_ops import cutlass_scaled_fp4_mm, fusedQuantizeNv
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
    CutlassNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    slice_nvfp4_output,
)
from vllm.platforms import current_platform

# Imported for its registration side effect: defines the
# ``vllm_omni::fused_hadamard_nvfp4_quant`` custom op (with a fake impl) so the
# fused path is a single opaque node under torch.compile instead of a graph
# break. The heavy JIT build stays lazy (only on first op call).
from vllm_omni.quantization import fused_hadamard_nvfp4  # noqa: F401

QutlassTransform = Literal["identity", "random_hadamard"]
_SUPPORTED_BLOCK_SIZES = {16, 32, 64, 128}
_TRANSFORM_CACHE: dict[tuple[str, int | None, torch.dtype, int, int, str], torch.Tensor] = {}


@dataclass(frozen=True)
class QutlassNvFp4Options:
    """Configuration for the opt-in QuTLASS NVFP4 path."""

    transform: QutlassTransform = "random_hadamard"
    block_size: int = 16
    seed: int = 0
    # When True (and block_size == 16 on SM100+/CUDA>=12.9), use the fused
    # in-register Had16 + NVFP4 quant kernel (vllm_omni.quantization.
    # fused_hadamard_nvfp4) + vLLM's cutlass_scaled_fp4_mm instead of QuTLASS's
    # fusedQuantizeNv + to_blocked + matmul_nvf4. Same math, ~pure-NVFP4 speed.
    fused_kernel: bool = True


def parse_qutlass_nvfp4_options(additional_config: dict[str, Any] | None) -> QutlassNvFp4Options | None:
    """Return validated options when the experimental path is enabled."""
    config = additional_config or {}
    enabled = config.get("qutlass_nvfp4", False)
    if not isinstance(enabled, bool):
        raise TypeError("additional_config.qutlass_nvfp4 must be a boolean")
    if not enabled:
        return None

    transform = config.get("qutlass_nvfp4_transform", "random_hadamard")
    if transform not in {"identity", "random_hadamard"}:
        raise ValueError(
            "additional_config.qutlass_nvfp4_transform must be "
            "'identity' or 'random_hadamard'"
        )

    block_size = config.get("qutlass_nvfp4_block_size", 16)
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise TypeError("additional_config.qutlass_nvfp4_block_size must be an integer")
    if block_size not in _SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            "additional_config.qutlass_nvfp4_block_size must be one of "
            f"{sorted(_SUPPORTED_BLOCK_SIZES)}"
        )

    seed = config.get("qutlass_nvfp4_seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("additional_config.qutlass_nvfp4_seed must be an integer")

    fused_kernel = config.get("qutlass_nvfp4_fused_kernel", True)
    if not isinstance(fused_kernel, bool):
        raise TypeError("additional_config.qutlass_nvfp4_fused_kernel must be a boolean")

    return QutlassNvFp4Options(
        transform=transform,
        block_size=block_size,
        seed=seed,
        fused_kernel=fused_kernel,
    )


def _normalized_hadamard(block_size: int) -> torch.Tensor:
    """Construct a normalized Sylvester Hadamard matrix on CPU."""
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < block_size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(block_size)


def get_qutlass_transform(
    *,
    device: torch.device,
    dtype: torch.dtype,
    options: QutlassNvFp4Options,
) -> torch.Tensor:
    """Return a cached identity or reproducibly randomized Hadamard transform."""
    key = (
        device.type,
        device.index,
        dtype,
        options.block_size,
        options.seed,
        options.transform,
    )
    cached = _TRANSFORM_CACHE.get(key)
    if cached is not None:
        return cached

    if options.transform == "identity":
        transform = torch.eye(options.block_size, dtype=torch.float32)
    else:
        transform = _normalized_hadamard(options.block_size)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(options.seed)
        signs = torch.randint(
            0,
            2,
            (options.block_size,),
            generator=generator,
            dtype=torch.int64,
        ).to(torch.float32)
        signs = signs.mul_(2).sub_(1)
        permutation = torch.randperm(options.block_size, generator=generator)
        transform = transform[:, permutation] * signs

    cached = transform.to(device=device, dtype=dtype).contiguous()
    _TRANSFORM_CACHE[key] = cached
    return cached


@torch.library.custom_op("vllm_omni::qutlass_nvfp4_quantize", mutates_args=())
def qutlass_nvfp4_quantize(
    activations: torch.Tensor,
    transform: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep QuTLASS's output-buffer quantizer opaque to torch.compile."""
    return fusedQuantizeNv(activations, transform, global_scale)


@qutlass_nvfp4_quantize.register_fake
def _fake_qutlass_nvfp4_quantize(
    activations: torch.Tensor,
    transform: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del transform, global_scale
    packed = activations.new_empty(
        (*activations.shape[:-1], activations.shape[-1] // 2),
        dtype=torch.uint8,
    )
    rows = activations.numel() // activations.shape[-1]
    cols = activations.shape[-1] // 16
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((cols + 3) // 4) * 4
    scales = activations.new_empty(
        (padded_rows, padded_cols),
        dtype=torch.float8_e4m3fn,
    )
    return packed, scales


@torch.library.custom_op("vllm_omni::qutlass_nvfp4_matmul", mutates_args=())
def qutlass_nvfp4_matmul(
    activations: torch.Tensor,
    weight: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Apply QuTLASS scale swizzling and its native NVFP4 GEMM."""
    blocked_activation_scales = to_blocked(
        activation_scales,
        backend="triton",
    )  # [M_sf*K_sf]
    return torch.ops._qutlass_C.matmul_nvf4_bf16_tn(
        activations,
        weight,
        blocked_activation_scales,
        weight_scales,
        alpha,
    )


@qutlass_nvfp4_matmul.register_fake
def _fake_qutlass_nvfp4_matmul(
    activations: torch.Tensor,
    weight: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    del activation_scales, weight_scales, alpha
    return activations.new_empty(
        (activations.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
    )  # [M,N]


class QutlassNvFp4LinearKernel(CutlassNvFp4LinearKernel):
    """QuTLASS fused transform/quantization plus vLLM's CUTLASS NVFP4 GEMM."""

    def __init__(
        self,
        config: NvFp4LinearLayerConfig,
        options: QutlassNvFp4Options,
    ) -> None:
        super().__init__(config)
        self.options = options
        self._signs: torch.Tensor | None = None

    def _fused_kernel_enabled(self) -> bool:
        """Whether the in-register fused Had16 + NVFP4 quant kernel is used.

        Requires block_size == 16 (NVFP4 group == Hadamard block == one PACK16
        thread's width) and a Hadamard transform -- the fused kernel always
        rotates, so ``identity`` and other block sizes fall back to QuTLASS
        ``fusedQuantizeNv``.
        """
        return (
            self.options.fused_kernel
            and self.options.block_size == 16
            and self.options.transform != "identity"
        )

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        cutlass_supported, reason = super().is_supported(compute_capability)
        if not cutlass_supported:
            return False, reason
        if not current_platform.is_cuda() or not current_platform.has_device_capability(100):
            return False, "QuTLASS NVFP4 requires an NVIDIA Blackwell GPU (SM100+)"
        # The fused block=16 path needs only the base CUTLASS NVFP4 GEMM; the
        # QuTLASS custom ops are required only for the non-fused fallback and
        # are validated in process_weights_after_loading when that path runs.
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare all static inputs once during model loading."""
        super().process_weights_after_loading(layer)

        if self._fused_kernel_enabled():
            # The fused path mirrors vLLM's CutlassNvFp4LinearKernel exactly
            # (same input_global_scale_inv / alpha / swizzled weight_scale),
            # only adding the in-register Had16 to the activation quant. No
            # random signs for now -> plain normalized Sylvester Had16; a
            # numerically-correct model needs its weights rotated with the same
            # transform offline (see module docstring).
            self._signs = None
            return

        # QuTLASS fallback (block_size != 16, or fused disabled). The QuTLASS
        # custom ops are exercised in apply_weights, not here.
        layer.input_global_scale_inv = torch.nn.Parameter(
            layer.input_global_scale_inv.detach().amax().reshape(1).contiguous(),
            requires_grad=False,
        )  # [1]
        transform = get_qutlass_transform(
            device=layer.weight.device,
            dtype=torch.bfloat16,
            options=self.options,
        )  # [R,R]
        if hasattr(layer, "register_buffer"):
            layer.register_buffer(
                "qutlass_transform",
                transform,
                persistent=False,
            )
        else:
            layer.qutlass_transform = transform

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply fused online transform/quantization and the NVFP4 GEMM."""
        if x.dtype != torch.bfloat16:
            raise TypeError(f"QuTLASS NVFP4 requires bfloat16 activations, got {x.dtype}")
        output_size = layer.output_size_per_partition
        output_shape = [*x.shape[:-1], output_size]
        weights_padding_bytes = getattr(layer, "weights_padding_cols", 0)

        x_flat = x.contiguous().flatten(end_dim=-2)  # [M,K]
        if weights_padding_bytes:
            x_flat = F.pad(x_flat, (0, weights_padding_bytes * 2))  # [M,K_padded]
        if x_flat.shape[-1] % self.options.block_size != 0:
            raise ValueError(
                f"activation width {x_flat.shape[-1]} is not divisible by "
                f"QuTLASS transform block size {self.options.block_size}"
            )

        if self._fused_kernel_enabled():
            # Call the registered custom op directly (no in-function import,
            # which would itself graph-break under torch.compile). The op is
            # registered at module import (see the fused_hadamard_nvfp4 import).
            x_fp4, x_scales = torch.ops.vllm_omni.fused_hadamard_nvfp4_quant(
                x_flat,
                layer.input_global_scale_inv,
                self._signs,
            )  # [M,K/2], swizzled SF
            out = cutlass_scaled_fp4_mm(  # [M,N_padded]
                x_fp4,
                layer.weight,
                x_scales,
                layer.weight_scale,
                layer.alpha,
                torch.bfloat16,
            )
        else:
            x_fp4, x_scales = qutlass_nvfp4_quantize(  # [M,K_packed], [M_sf,K_sf]
                x_flat,
                layer.qutlass_transform,
                layer.input_global_scale_inv,
            )
            out = qutlass_nvfp4_matmul(
                x_fp4,
                layer.weight,
                x_scales,
                layer.weight_scale,
                layer.alpha,
            )  # [M,N_padded]

        out = slice_nvfp4_output(out, output_size)  # [M,N]

        if bias is not None:
            out = out + bias  # [M,N]
        return out.view(*output_shape)
