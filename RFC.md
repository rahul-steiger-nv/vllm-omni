# RFC: Standardizing Diffusion Attention Execution Contracts

## Motivation

### 1. Overview

This RFC extends the existing diffusion attention selection contract with a common runtime capability, lifecycle, and compilation contract. Existing backends will migrate incrementally with conservative defaults and without changing their behavior.

#### Relationship to existing design

[`attention_backend_selection.md`](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/attention_backend_selection.md) already defines configuration resolution, the registry/platform boundary, typed backend options on `AttentionSpec`, and the checklist for adding or changing a backend. Those responsibilities remain unchanged: `AttentionSpec` owns user configuration, the registry maps stable names to classes, and `OmniPlatform` owns dependency checks, hardware compatibility, and automatic selection.

#### Problem

Selection is documented, but post-selection requirements remain spread across attention, backend, model, and parallelism code. Existing class methods cover some individual capabilities, while other compatibility checks still depend on backend names or are enforced only inside execution paths. There is no common way to describe interactions among masks, packed inputs, paged KV, fallback, parallel strategies, and compilation.

An existing [`TRTLLM_ATTN` versus AllGather-KV TODO](https://github.com/vllm-project/vllm-omni/blob/7be014bc/vllm_omni/diffusion/attention/layer.py#L162-L168) asks to replace a backend-name check with an `AttentionBackend` capability.

<details>
<summary>Additional code evidence</summary>

- [`vllm_omni/diffusion/attention/parallel/ring.py:185-207`](https://github.com/vllm-project/vllm-omni/blob/7be014bc/vllm_omni/diffusion/attention/parallel/ring.py#L185-L207), where `_non_ring_prefs` enumerates backends that Ring cannot execute directly;
- FlashInfer defines an [SDPA fallback](https://github.com/vllm-project/vllm-omni/blob/7be014bc/vllm_omni/diffusion/attention/backends/flashinfer_attn.py#L359-L378) gated by explicit selection at [three call sites](https://github.com/vllm-project/vllm-omni/blob/7be014bc/vllm_omni/diffusion/attention/backends/flashinfer_attn.py#L408-L439), while FastVideo independently derives [runtime fallback reasons](https://github.com/vllm-project/vllm-omni/blob/7be014bc/vllm_omni/diffusion/attention/backends/fastvideo_vsa.py#L453-L504).

</details>

Compilation handling is also inconsistent. Some kernel paths are traced directly, some use `torch.compiler.disable`, and others use `torch.library.custom_op`. FlashInfer ([#4989](https://github.com/vllm-project/vllm-omni/pull/4989)), TRTLLM ([#7214](https://github.com/vllm-project/vllm-omni/pull/7214)), and FlashAttention-4 ([#7220](https://github.com/vllm-project/vllm-omni/pull/7220)) have all required separate fixes for similar compiler-boundary problems.

## Proposed Change

### 2. Scope & Objectives

#### Goals

- Replace backend-name checks with discoverable, path-specific capability results and actionable errors.
- Standardize initialization, shape-dependent planning, and compiler boundaries for stateful and stateless backends.
- Avoid device synchronization and per-denoising-step Python overhead by resolving capabilities during selection, initialization, or planning.

#### Non-goals

- Forcing every backend or execution path to support fullgraph compilation.
- Defining one universal attention custom operator.
- Unifying diffusion and autoregressive attention.

### 3. Design

#### Runtime capability contract

Each backend resolves capabilities for an execution context that may include the platform, kernel variant, dtype, mask or packing mode, paged-KV mode, parallel strategy, and outer boundaries such as HSDP. Capabilities are path-specific because a single backend class may dispatch to implementations with different support.

`AttentionBackend` provides conservative pre-construction defaults, while `AttentionImpl` resolves path-specific capabilities after kernel selection; `OmniPlatform` remains responsible for consuming pre-construction facts and enforcing selection policy.

Each result is `SUPPORTED`, `UNSUPPORTED`, or `UNMIGRATED`. `UNSUPPORTED` is a verified rejection with an actionable reason; `UNMIGRATED` preserves existing behavior while a path has no verified declaration.

A path reports `SUPPORTED` only when its normalized `(platform, kernel, dtype, causal mode, metadata mode, parallel strategy, outer boundary)` combination is explicitly allowlisted by tests; every other combination defaults to `UNMIGRATED`. Required predicates that are unknown before runtime, such as whether packed NPU metadata satisfies the varlen contract, also remain `UNMIGRATED`.

A resolved result describes the complete `Attention.forward` path: platform kernel, parallel strategy, fallback, outer compilation boundaries, semantic guarantees, and planning policy. Parallel compatibility distinguishes wrapping local attention from implementing a parallel kernel directly. Guarantees use a typed enum rather than strings; fallback policy is `AUTO_ONLY`, `EXPLICIT_ALLOWED`, or `NEVER`.

Planning and execution use the same normalization function. Its plan key contains only the normalized subset that affects dispatch or planning, including producer-published semantic mask/packing modes and metadata revisions. Unknown content-sensitive metadata is volatile: capability resolution returns an uncached, conservative result without reading device values. Existing runtime dispatch may still inspect those values on unmigrated paths.

The intended plan owner is the caller that prepares attention inputs before denoising execution. It resolves the complete path after backend initialization, validates request-level guarantees, and caches a plan only when the normalized key is non-null. Before reuse, that caller checks dispatch-relevant shapes, semantic modes, metadata revisions, and parallel/outer-boundary configuration. Producers must update the revision whenever content that affects a plan changes; absent that contract, the caller cannot cache content-dependent plans.

For example, two batches can both have sequence length 128 while their document boundaries change from `[0, 64, 128]` to `[0, 32, 128]`. The producer publishes a new metadata revision, so the preparation caller discards the previous plan and resolves again before execution. Without a trusted revision, packed metadata remains volatile. This lifecycle is proposed; the current PoC returns planning keys but does not install a plan cache or preparation caller.

<details>
<summary>Cross-platform example</summary>

For MiniMax-H3 packed attention, `FLASH_ATTN` on ROCm consumes arbitrary multi-document `cu_seqlens` through AITER, while NPU MindIE accepts only the `[real, pad]` packed contract and may rebuild a mask or use prefix slicing. The same backend name therefore resolves different packed-input, fallback, parallel, and compilation guarantees by platform and metadata.

Worked example (proposed migration, with BF16, non-causal self-attention and no sequence parallelism):

| Input and platform | Path to resolve | Required semantic check |
| --- | --- | --- |
| ROCm/AITER, three real documents with `cu_seqlens=[0, 48, 96, 128]` | Packed varlen | Each document is isolated; output matches concatenated per-document attention. |
| NPU/MindIE, one real prefix and padding with `cu_seqlens=[0, 96, 128]`, `valid_kv_length=96`, and `npu_attn_varlen=True` | Packed varlen, or prefix slicing when `MINDIE_SD_FA_TYPE=ascend_laser_attention` | Valid queries cannot attend to padding; compare valid-prefix outputs with the dense masked reference. |
| NPU/MindIE, three real documents with `cu_seqlens=[0, 48, 96, 128]` | The two-document fast path cannot provide the requested isolation | A padding-only fallback is insufficient. A migrated caller must reject this request unless a separately verified fallback preserves all document boundaries. |

All these paths remain `UNMIGRATED` in this PoC, with no guaranteed-fullgraph claim. A future migrated result identifies the selected kernel and fallback, its explicit-selection policy, and the guarantees actually validated. The NPU environment-selected kernel belongs in initialization state and the plan key; changing it requires reinitialization or invalidation. Adding Ring, Ulysses, or HSDP requires resolving the composed path again rather than carrying over the local-kernel declaration.

Platform validation runs these cases on their respective hardware, then changes boundaries at the same total length and checks re-resolution and numerical equivalence. NPU tests distinguish rejection of the fast-path contract from rejection of the complete request: a fallback is acceptable only if it preserves the requested semantics. Compilation tests must exercise the complete selected path before upgrading its declaration.

</details>

<details>
<summary>Existing adapter coverage</summary>

- `accept_output_buffer`, `supports_piecewise_spans`, `supports_paged_kv`, and `supports_prefix_kv_slicing`
- `supports_packed_mask_free()`, `supports_multi_doc_packed_varlen()`, and `supports_attention_mask()`
- `supported_platforms`, `validate_available()`, `get_supported_head_sizes()`, and `supports_head_size()`
- `indexes_kv_by_block_stride()` and `AttentionImpl.supports_kv_cache_dtype()`

</details>

#### Compilation contract

The complete resolved path reports the most restrictive compilation mode across the backend kernel and outer boundaries:

- `TRACEABLE`: safe to trace directly.
- `CUSTOM_OP`: execution stays opaque and provides a fake implementation.
- `EAGER_ONLY`: fullgraph compilation is unsupported.

The mode may differ across paths in the same backend, such as dense, masked, packed-varlen, fallback, or platform-specific implementations. Using `torch.compiler.disable` inside a `CUSTOM_OP` implementation is valid because the operator is already opaque to the caller's graph. An outer disabled boundary, including the HSDP boundary in `Attention.forward`, makes the complete path `EAGER_ONLY` even when its backend kernel is `CUSTOM_OP`.

At the request-validation boundary, a guaranteed-fullgraph request accepts only a complete path with verified `TRACEABLE` or `CUSTOM_OP` compilation; `EAGER_ONLY` and `UNMIGRATED` paths are rejected before execution.

#### Lifecycle and custom-op infrastructure

Backends separate initialization and dependency loading, shape-dependent planning, and tensor execution. Imports, file I/O, JIT loading, Python caches, and planning objects must not accidentally enter a compiled graph.

For `CUSTOM_OP` paths, shared infrastructure standardizes registration, state lookup, lifetime errors, and idempotent module loading. Each backend still owns its schema, fake implementation, mutation declaration, and kernel behavior. Stateful paths declare whether state is process-, device-, or instance-owned, keep handles alive for every compiled graph that references them, and avoid Python-value guards that prevent graph reuse for equivalent instances.

For a migrated stateful path, the graph/execution owner retains a strong lease on referenced state until all dependent graphs and in-flight executions are released. Destroying the originating backend instance must not invalidate a live graph's lease; explicit use of an expired handle raises an actionable error. State may be shared across equivalent instances only when device, kernel configuration, and planning requirements match. Mutable planning buffers must be isolated per execution or protected against concurrent reuse. The concrete lease/lookup API is deferred to the TRTLLM PoC.

A compilation declaration applies to a documented shape/layout domain, not arbitrary tensor shapes. `dynamic=True` alone is not evidence of reuse: tests must count compilations across supported shape changes, replay graphs after backend-instance turnover, and check numerical results. Shape changes outside that domain require re-resolution and may create another graph. Eager planning and CUDA graph replay are separate concerns; this RFC's fullgraph declaration describes `torch.compile`, and does not by itself promise CUDA graph capture compatibility.

#### Fallback contract

Every fallback documents:

- its typed policy (`AUTO_ONLY`, `EXPLICIT_ALLOWED`, or `NEVER`);
- which unsupported inputs trigger it; and
- whether it preserves masks, packed boundaries, causality, dtype, and compilation guarantees.

A backend must not advertise a capability provided only by a semantically weaker fallback. Explicit selections fail unless the fallback policy is `EXPLICIT_ALLOWED`.

#### Dependencies, risks, and mitigations

- Incorrect declarations could admit unsupported combinations. Each migrated path therefore tests both supported and rejected contexts.
- Stateful handles may violate lifetime or graph-reuse guarantees; shared lookup and multi-instance replay tests mitigate this.

#### Migration

1. Introduce the capability and compilation contracts as adapters over existing class methods. Unmigrated paths report `UNMIGRATED`; only verified `UNSUPPORTED` results may reject execution. During migration results are advisory until the relevant caller and path have migrated.
2. Start with dense BF16 FA4 on CUDA. ROCm, NPU, packed, fallback, and parallel paths remain `UNMIGRATED` until their semantic validation passes; then migrate one verified path at a time.
3. Add shared state-lookup infrastructure when migrating the first backend that requires it; the stateless FA4 compiler fix remains independent of this RFC.
4. Replace backend-name checks only after the corresponding complete paths and callers are verified.

#### Current proof-of-concept boundary

Implemented: tri-state result types, request-level support validation, metadata normalization shared by FA dispatch and resolution, a dense non-causal BF16 CUDA/FA4 declaration, and inspection of parallel and HSDP boundaries through `Attention.resolve_execution_path`. Tests distinguish mocked compiler-boundary coverage from real FA4 numerical validation.

Deferred: a production caller for request enforcement, a user-facing guaranteed-fullgraph option, plan caching and producer revision integration, parallel/fallback composition for migrated paths, and stateful custom-op infrastructure. Ordinary execution remains advisory: calling `resolve_execution_path` reports a path, while `requested_support` evaluates an explicit request without altering that path's migration state. The future preparation caller must reject an unsupported request before invoking execution.

FA4 demonstrates a stateless migration only. A separate TRTLLM PoC must demonstrate state ownership, handle lifetime, dynamic-shape planning, and equivalent-instance graph reuse before the stateful infrastructure can be considered validated.

### 4. Correctness & Testing Plans

Declared capabilities must match backend behavior, while migration preserves outputs and fallback semantics.

- Adapter and runtime tests cover all three support states, explicit verified-path allowlists, complete-path guarantees, and dispatch changes caused by same-shape metadata.
- ROCm validation compares multi-document packed `FLASH_ATTN` output with separate per-document attention calls.
- NPU validation covers the supported `[real, pad]` contract, rejects arbitrary multi-document packing, and compares masked and prefix-slicing fallbacks with their dense reference.
- Both platforms test fixed-shape metadata changes and verify that the resolved compilation mode matches the path actually executed.
- `CUSTOM_OP` paths keep a fast mocked Dynamo boundary test and add an Inductor test through `Attention.forward`; HSDP tests verify that its outer boundary downgrades the complete path to `EAGER_ONLY`.
- Hardware validation includes a real Blackwell FA4 smoke test. Stateful paths cover multiple dynamic shapes and backend instances without unintended recompilation.
- Custom-op schemas preserve mutation and aliasing declarations; fake implementations preserve output shape, dtype, and device.

### 5. Open Questions & Discussions

- Where should stateful custom-op support live?
