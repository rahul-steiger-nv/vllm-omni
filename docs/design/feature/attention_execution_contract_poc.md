# Diffusion attention execution contract PoC

This prototype accompanies [RFC #7226](https://github.com/vllm-project/vllm-omni/issues/7226)
and extends the [attention selection design](attention_backend_selection.md).
It demonstrates path-specific capabilities with dense BF16 FA4 and tensor-state
lifetime with a test-only attention module.

## Execution contract

`ExecutionContext` describes the requested execution path. `ExecutionPathResult`
reports its identity, support status and reason, and compilation mode.
`AttentionBackend.resolve_capabilities()` provides conservative pre-construction
results. After initialization, `Attention.resolve_execution_path()` supplies the
active parallel, paged-KV, and HSDP context; the backend combines it with the
selected kernel and normalized metadata.

Resolution runs outside compiled execution. FA4 reports `SUPPORTED` and
`CUSTOM_OP` for dense, noncausal BF16 without parallel or HSDP boundaries when
its kernel accepts the head dimensions. Dimension validation delegates to FA4's
architecture-specific rules through `backends/utils/fa.py`. Kernel rejections
report `UNSUPPORTED` with an actionable reason. Missing private validators and
other unmigrated paths report `UNMIGRATED` with advisory `EAGER_ONLY` defaults.
`requested_support()` checks a fullgraph request; it does not enforce selection
or change existing execution.

FA4 execution uses an opaque custom op. Its fake output preserves Q's batch,
sequence, and head count and uses V's head dimension. Metadata normalization is
shared with dispatch. Producers must update published mask semantics when masks
change; unpublished masks remain runtime-dependent to avoid synchronization.
Callers resolve again when execution metadata changes.

## State lifetime example

`test_attention_state_lifetime.py` prepares an owned query-scale tensor eagerly
and passes it explicitly to an opaque attention op. The compiled module retains
its state after the caller drops its reference and releases ownership when the
compiled callable is deleted. Equivalent instances reuse a graph with their own
state values.

This example tests ownership and release, dynamic replay across instances,
custom-op schema/fake behavior, and Inductor execution. It adds no production
backend or state registry. Planning keys, caching, fallback-policy declarations,
opaque handles, CUDA graph capture, and TRTLLM integration are deferred.

## Validation

```bash
python -m pytest tests/diffusion/attention/test_flash_attn_compile.py \
  tests/diffusion/attention/test_flash_attn.py \
  tests/diffusion/attention/test_attention_capabilities.py \
  tests/diffusion/attention/test_attention_state_lifetime.py -q -rs
```

The focused suite covers capability decisions and metadata changes, existing
padding/mask regressions, tensor-state lifetime, and real FA4 fullgraph execution
against FP32 SDPA and eager FA4. Representative Q/K and V dimensions are
(32, 32), (64, 64), (80, 48), (192, 128), and (256, 256), with batches 1 and 2
and sequence lengths up to 1,024. Invalid dimensions are checked against actual
kernel errors. These samples exercise the contract rather than define support.
BF16 comparisons use `atol=rtol=1e-2`; `torch.library.opcheck` checks schema and
fake-output correctness.

Single-graph reuse is asserted in the tensor-state test. The FA4 numerical tests
retain compiler caches across shapes within each case but do not require one
graph across different batch sizes, Q/K length equality, or head dimensions.

Tested environment: GB300, PyTorch `2.13.0+cu130`, FA4 `4.0.0b18`, CUTLASS DSL
`4.6.2` with CUDA 13 libraries, Quack `0.6.4`, and TVM FFI `0.1.11`.
Real-kernel tests require CUDA and CuTe FA4; their availability is checked at runtime.

**86 passed, no skips** on 2026-09-11 (47.38 seconds of test execution).
Ruff lint and formatting pass. Upstream PyTorch/CUTLASS warnings remain.
