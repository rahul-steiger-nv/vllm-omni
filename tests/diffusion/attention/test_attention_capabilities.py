# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm_omni.diffusion.attention.capabilities import (
    CapabilityResult,
    CompilationMode,
    ExecutionContext,
    OuterBoundary,
    ParallelStrategy,
    StateOwnership,
    SupportStatus,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _impl(*, kernel_variant: str | None = "fa4", causal: bool = False):
    impl = FlashAttentionImpl.__new__(FlashAttentionImpl)
    impl._kernel_variant = kernel_variant
    impl.causal = causal
    return impl


def _resolve(
    impl,
    *,
    context: ExecutionContext | None = None,
    attn_metadata: AttentionMetadata | None = None,
    dtype=torch.bfloat16,
):
    tensor = torch.empty((1, 16, 8, 64), dtype=dtype)
    return impl.resolve_execution_path(
        context or ExecutionContext(platform="cuda"),
        tensor,
        tensor,
        tensor,
        attn_metadata,
    )


class _UnmigratedBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "UNMIGRATED_TEST"

    @staticmethod
    def get_impl_cls():
        return None

    @staticmethod
    def get_metadata_cls():
        return None

    @staticmethod
    def get_builder_cls():
        return None

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []


def test_capability_result_is_tri_state():
    assert CapabilityResult.supported().status is SupportStatus.SUPPORTED
    assert CapabilityResult.unsupported("not available").status is SupportStatus.UNSUPPORTED
    assert CapabilityResult.unmigrated().status is SupportStatus.UNMIGRATED
    with pytest.raises(ValueError, match="actionable reason"):
        CapabilityResult(SupportStatus.UNSUPPORTED)


def test_unmigrated_backend_default_is_conservative():
    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    result = _UnmigratedBackend.resolve_capabilities(context)

    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.compilation_mode is CompilationMode.EAGER_ONLY
    requested = result.requested_support(context)
    assert requested.status is SupportStatus.UNSUPPORTED
    assert "no verified fullgraph declaration" in requested.reason
    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.requested_support(replace(context, require_fullgraph=False)).status is SupportStatus.UNMIGRATED


@pytest.mark.parametrize("mode", list(CompilationMode))
def test_fullgraph_request_checks_compilation_mode(mode):
    result = replace(_resolve(_impl()), compilation_mode=mode)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    expected = SupportStatus.UNSUPPORTED if mode is CompilationMode.EAGER_ONLY else SupportStatus.SUPPORTED
    assert result.requested_support(context).status is expected
    assert result.requested_support(replace(context, require_fullgraph=False)).status is SupportStatus.SUPPORTED


def test_request_preserves_verified_rejection_reason():
    result = replace(_resolve(_impl()), support=CapabilityResult.unsupported("unsupported head size"))
    assert result.requested_support(ExecutionContext(platform="cuda", require_fullgraph=True)) == result.support


def test_planning_key_uses_normalized_fields_and_metadata_revision():
    first = ExecutionContext(
        platform="cuda",
        shape_signature=(1, 16, 8, 64),
        metadata_revision=1,
    )
    changed = replace(first, metadata_revision=2)

    assert first.make_planning_key("cuda", "fa4", "fa4_dense") != changed.make_planning_key("cuda", "fa4", "fa4_dense")
    assert replace(first, volatile_metadata=True).make_planning_key("cuda", "fa4", "fa4_dense") is None


def test_exact_cuda_fa4_dense_path_is_supported():
    context = ExecutionContext(
        platform="cuda",
        require_fullgraph=True,
    )
    result = _resolve(_impl(), context=context)

    assert result.path == "fa4_dense"
    assert result.support.status is SupportStatus.SUPPORTED
    assert result.compilation_mode is CompilationMode.CUSTOM_OP
    assert result.state_ownership is StateOwnership.STATELESS
    assert result.requested_support(context).status is SupportStatus.SUPPORTED
    assert result.planning_key is not None


def test_causal_fa4_path_remains_unmigrated():
    result = _resolve(_impl(causal=True))

    assert result.support.status is SupportStatus.UNMIGRATED


def test_initialized_kernel_identity_overrides_caller_claim():
    result = _resolve(
        _impl(kernel_variant=None),
        context=ExecutionContext(
            platform="cuda",
            kernel_variant="fa4",
        ),
    )

    assert result.support.status is SupportStatus.UNMIGRATED


def test_preconstruction_without_resolved_cuda_kernel_is_unmigrated():
    result = FlashAttentionBackend.resolve_capabilities(ExecutionContext(platform="cuda"))

    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.path == "unverified"


@pytest.mark.parametrize(
    "changes",
    [
        {"platform": "rocm"},
        {"paged_kv": True},
        {"parallel_strategy": ParallelStrategy.ULYSSES},
        {"parallel_strategy": ParallelStrategy.RING},
        {"parallel_strategy": ParallelStrategy.HYBRID_ULYSSES_RING},
        {"outer_boundaries": frozenset({OuterBoundary.HSDP})},
    ],
)
def test_unverified_flash_attention_variants_remain_unmigrated(changes):
    result = _resolve(
        _impl(),
        context=replace(ExecutionContext(platform="cuda"), **changes),
    )

    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.compilation_mode is CompilationMode.EAGER_ONLY
    assert result.state_ownership is None


def test_unverified_dtype_remains_unmigrated():
    result = _resolve(_impl(), dtype=torch.float32)

    assert result.support.status is SupportStatus.UNMIGRATED


@pytest.mark.parametrize(
    "attn_metadata",
    [
        AttentionMetadata(full_attn_spans=[[(0, 8)]]),
        AttentionMetadata(
            extra={
                "cu_seqlens_q": torch.tensor([0, 16], dtype=torch.int32),
                "cu_seqlens_k": torch.tensor([0, 16], dtype=torch.int32),
                "max_seqlen_q": 16,
                "max_seqlen_k": 16,
            }
        ),
        AttentionMetadata(extra={"kv_cache_dtype": "fp8"}),
    ],
)
def test_metadata_normalization_prevents_dense_support(attn_metadata):
    result = _resolve(_impl(), attn_metadata=attn_metadata)

    assert result.support.status is SupportStatus.UNMIGRATED


def test_unpublished_all_true_mask_remains_runtime_dependent():
    metadata = AttentionMetadata(
        attn_mask=torch.ones((1, 16), dtype=torch.bool),
    )
    result = _resolve(_impl(), attn_metadata=metadata)

    assert result.path == "runtime_mask_dependent"
    assert result.support.status is SupportStatus.UNMIGRATED


def test_producer_published_noop_mask_resolves_dense_fa4():
    metadata = AttentionMetadata(
        attn_mask=torch.ones((1, 16), dtype=torch.bool),
        extra={"attention_mask_mode": "none"},
    )
    result = _resolve(_impl(), attn_metadata=metadata)

    assert result.path == "fa4_dense"
    assert result.support.status is SupportStatus.SUPPORTED


def test_fixed_shape_mask_semantic_change_invalidates_dense_resolution():
    metadata = AttentionMetadata(
        attn_mask=torch.ones((1, 16), dtype=torch.bool),
        extra={"attention_mask_mode": "none"},
    )
    impl = _impl()
    dense = _resolve(impl, attn_metadata=metadata)
    assert dense.planning_key is not None

    metadata.attn_mask[:, 8:] = False
    metadata.extra["attention_mask_mode"] = "padding"
    masked = _resolve(impl, attn_metadata=metadata)
    assert masked.support.status is SupportStatus.UNMIGRATED
    assert masked.planning_key is None
    request = ExecutionContext(platform="cuda", require_fullgraph=True)
    assert masked.requested_support(request).status is SupportStatus.UNSUPPORTED


def test_piecewise_dispatch_ignores_unused_incomplete_packed_metadata(monkeypatch):
    from vllm_omni.diffusion.attention.backends import flash_attn
    from vllm_omni.diffusion.attention.backends.utils import fa

    metadata = AttentionMetadata(full_attn_spans=[[(0, 16)]], extra={"max_seqlen_q": 16})
    impl = _impl()
    impl.fa_deterministic = False
    impl.softmax_scale = 0.125
    assert _resolve(impl, attn_metadata=metadata).support.status is SupportStatus.UNMIGRATED

    query = torch.empty((1, 16, 8, 64), dtype=torch.bfloat16)
    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(fa, "flash_attn_func", lambda *args, **kwargs: query)
    monkeypatch.setattr(flash_attn, "piecewise_attn", lambda *args, **kwargs: query)
    assert impl.forward_cuda(query, query, query, metadata) is query
