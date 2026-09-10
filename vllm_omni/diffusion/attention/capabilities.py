# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNMIGRATED = "unmigrated"


class CompilationMode(str, Enum):
    TRACEABLE = "traceable"
    CUSTOM_OP = "custom_op"
    EAGER_ONLY = "eager_only"


class PackingMode(str, Enum):
    NONE = "none"
    PACKED_PADDING = "packed_padding"
    MULTI_DOCUMENT = "multi_document"


class MaskMode(str, Enum):
    NONE = "none"
    PADDING = "padding"
    ARBITRARY = "arbitrary"
    UNKNOWN = "unknown"


class ParallelStrategy(str, Enum):
    NONE = "none"
    ULYSSES = "ulysses"
    RING = "ring"
    HYBRID_ULYSSES_RING = "hybrid_ulysses_ring"
    ALLGATHER_KV = "allgather_kv"


class OuterBoundary(str, Enum):
    HSDP = "hsdp"


class StateOwnership(str, Enum):
    STATELESS = "stateless"
    PROCESS = "process"
    DEVICE = "device"
    INSTANCE = "instance"


class SemanticGuarantee(str, Enum):
    CAUSALITY = "causality"
    DTYPE = "dtype"
    KV_BLOCK_STRIDE = "kv_block_stride"
    MASK = "mask"
    OUTPUT_BUFFER = "output_buffer"
    PACKED_BOUNDARIES = "packed_boundaries"
    PAGED_KV = "paged_kv"
    PIECEWISE = "piecewise"
    PREFIX_KV_SLICING = "prefix_kv_slicing"


class FallbackPolicy(str, Enum):
    AUTO_ONLY = "auto_only"
    EXPLICIT_ALLOWED = "explicit_allowed"
    NEVER = "never"


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    status: SupportStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is SupportStatus.UNSUPPORTED and not self.reason:
            raise ValueError("UNSUPPORTED capability results require an actionable reason")

    @classmethod
    def supported(cls) -> CapabilityResult:
        return cls(SupportStatus.SUPPORTED)

    @classmethod
    def unsupported(cls, reason: str) -> CapabilityResult:
        return cls(SupportStatus.UNSUPPORTED, reason)

    @classmethod
    def unmigrated(cls, reason: str | None = None) -> CapabilityResult:
        return cls(SupportStatus.UNMIGRATED, reason)


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    platform: str
    kernel_variant: str | None = None
    dtype: str | None = None
    causal: bool | None = None
    mask_mode: MaskMode = MaskMode.NONE
    packing_mode: PackingMode = PackingMode.NONE
    packed_contract_valid: bool | None = None
    piecewise: bool = False
    paged_kv: bool = False
    kv_cache_dtype: str | None = None
    parallel_strategy: ParallelStrategy = ParallelStrategy.NONE
    backend_explicit: bool = False
    require_fullgraph: bool = False
    shape_signature: tuple[Hashable, ...] = ()
    metadata_revision: Hashable | None = None
    volatile_metadata: bool = False
    outer_boundaries: frozenset[OuterBoundary] = frozenset()

    def make_planning_key(
        self,
        *normalized_execution_fields: Hashable,
    ) -> tuple[Hashable, ...] | None:
        """Build a key from the normalized subset that affects execution."""
        if self.volatile_metadata:
            return None
        return (
            *normalized_execution_fields,
            self.dtype,
            self.shape_signature,
            self.metadata_revision,
        )


@dataclass(frozen=True, slots=True)
class FallbackDescriptor:
    path: str
    reason: str
    policy: FallbackPolicy
    guarantees: frozenset[SemanticGuarantee] = frozenset()


@dataclass(frozen=True, slots=True)
class ExecutionPathResult:
    backend: str
    path: str
    support: CapabilityResult
    compilation_mode: CompilationMode
    platform: str
    kernel_variant: str | None
    parallel_strategy: ParallelStrategy
    guarantees: frozenset[SemanticGuarantee] = frozenset()
    fallback: FallbackDescriptor | None = None
    planning_key: tuple[Hashable, ...] | None = None
    state_ownership: StateOwnership | None = StateOwnership.STATELESS

    @classmethod
    def unmigrated(
        cls,
        backend: str,
        context: ExecutionContext,
        *,
        path: str = "unmigrated",
        guarantees: frozenset[SemanticGuarantee] = frozenset(),
    ) -> ExecutionPathResult:
        return cls(
            backend=backend,
            path=path,
            support=CapabilityResult.unmigrated(),
            compilation_mode=CompilationMode.EAGER_ONLY,
            platform=context.platform,
            kernel_variant=context.kernel_variant,
            parallel_strategy=context.parallel_strategy,
            guarantees=guarantees,
            planning_key=None,
            state_ownership=None,
        )

    def requested_support(self, context: ExecutionContext) -> CapabilityResult:
        """Validate request-level guarantees without changing path migration state."""
        if context.require_fullgraph and self.support.status is SupportStatus.UNMIGRATED:
            return CapabilityResult.unsupported(
                f"{self.backend} path {self.path!r} has no verified fullgraph declaration; "
                "use a verified path or disable the guaranteed-fullgraph request"
            )
        if self.support.status is not SupportStatus.SUPPORTED:
            return self.support
        if context.require_fullgraph and self.compilation_mode is CompilationMode.EAGER_ONLY:
            return CapabilityResult.unsupported(
                f"{self.backend} path {self.path!r} does not support guaranteed fullgraph compilation"
            )
        return CapabilityResult.supported()
