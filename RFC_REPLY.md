# Draft RFC replies

Do not post until the local RFC and PR #7220 changes have been reviewed and pushed.

## Reply to MaciejBalaNV

Thanks — agreed that a proof of concept will make the migration concrete. I am keeping [PR #7220](https://github.com/vllm-project/vllm-omni/pull/7220) focused on the independent FA4 compiler bugfix and preparing the capability proof of concept separately.

I propose `FLASH_ATTN` rather than TRTLLM for the first migration because the same backend name eventually exercises CUDA/FA4, ROCm/AITER, and NPU/MindIE paths. The initial PoC will deliberately verify only dense BF16 FA4; ROCm, NPU, packed, fallback, and parallel paths will remain `UNMIGRATED` until semantic validation exists.

This stateless PoC only partially answers the requested TRTLLM example. A follow-up TRTLLM PoC must exercise state ownership, handle lifetime, dynamic-shape planning, and graph reuse across equivalent instances, alongside its packed, SAGE fallback, and parallel constraints. FA4 alone does not validate the stateful infrastructure.

## Reply to hsliuustc0106

Thanks — I updated the proposed contract to address each point:

1. Resolve after backend initialization and before execution, with the input-preparation caller owning plan reuse. Dispatch-relevant shapes, semantic metadata revisions, and parallel/outer boundaries invalidate plans. A worked example changes document boundaries at fixed sequence length; unversioned content-sensitive metadata cannot be cached.
2. Results describe the complete path, including the selected kernel, parallel strategy, fallback policy, and typed semantic guarantees. An outer disabled boundary such as HSDP prevents a fullgraph guarantee.
3. Compilation guarantees cover a documented shape domain and require compilation-count and numerical tests. Stateful migrations must retain graph-owned state leases across backend-instance destruction and define concurrent use of mutable planning buffers. The concrete state API remains for the TRTLLM PoC.
4. `UNMIGRATED` preserves ordinary execution; verified `UNSUPPORTED` results reject migrated requests. Explicit guaranteed-fullgraph requests also reject unverified paths, without changing their migration status.
5. The worked non-CUDA example follows ROCm/AITER multi-document packing and NPU/MindIE `[real, pad]` packing through resolution. It explains why a padding-only fallback cannot preserve arbitrary document isolation. Platform validation covers per-document references, NPU prefix/masked equivalence, fixed-shape metadata changes, and compilation of the selected complete path.

The PoC implements resolution and request-validation APIs for dense non-causal BF16 FA4. Production enforcement, plan caching, producer revision integration, and stateful infrastructure remain deferred. The independent FA4 compiler fix stays separate; the broader lifecycle and non-CUDA examples specify requirements for subsequent migrations.
