## Purpose

Proof of concept for [RFC #7226](https://github.com/vllm-project/vllm-omni/issues/7226), validating execution capability declarations for dense, non-causal BF16 FA4 on CUDA. Includes the FA4 compiler fix from [#7220](https://github.com/vllm-project/vllm-omni/pull/7220) as a worked example; that fix remains independently reviewable there.

Adds tri-state support results, compilation declarations, shared metadata normalization, and complete-path resolution through `Attention`, including parallel and HSDP boundaries. Request validation rejects unverified fullgraph guarantees while ordinary execution preserves unmigrated behavior.

ROCm/NPU, packed, fallback, and parallel paths remain `UNMIGRATED`. Production request enforcement, plan caching, and stateful TRTLLM integration are deferred.

## Test Plan

Requires CUDA/Blackwell and `flash-attn-4[cu13]==4.0.0b18` for real-kernel validation.

```bash
pytest -q tests/diffusion/attention/test_{attention_capabilities,flash_attn_compile,flash_attn}.py
# CI-style selection for the real FA4 test:
pytest -q tests/diffusion/attention/test_flash_attn_compile.py -m 'core_model and cuda and B200' --run-level core_model
```

**vLLM Version:** 0.28.0

**vLLM-Omni Commit:** `f10a99e0` plus this PoC

## Test Result

- 67 capability, compile, and FlashAttention tests passed on GB300.
- Real FA4 matched FP32 math SDPA through fullgraph Inductor on GB300 at sequence lengths 16, 24, and 48.
- Fixed-shape mask changes invalidate dense-path resolution.
- Applicable pre-commit checks passed except three existing mypy errors in `layer.py`, reproduced on the unchanged #7220 base. Diff checks passed.
