// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused block-Hadamard(16) + NVFP4 quantization for the MR-GPTQ activation
// path (arXiv:2509.23202). This is a drop-in replacement for
// vllm::scaled_fp4_quant that folds the online block-diagonal Hadamard rotation
// into the quantization kernel, so the rotation costs one extra in-register
// 16x16 matvec instead of QuTLASS's fusedQuantizeNv CUTLASS GEMM (+ per-call
// cudaMalloc/initialize) followed by a separate to_blocked swizzle.
//
// The NVFP4 group size (16) equals the Hadamard block size (16), so one thread
// owns a full group and the rotation is entirely local: no warp shuffles, no
// GEMM machinery, no separate scale-swizzle launch. The scale-factor output is
// written in the exact swizzled tcgen05 layout that cutlass_scaled_fp4_mm
// expects, identical to vLLM's own scaled_fp4_quant(is_sf_swizzled_layout=True).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <optional>
#include <tuple>

// Enable the 16-elements-per-thread (PACK16) path; requires CUDA >= 12.9.
#define NVFP4_ENABLE_ELTS16 1
#include "nvfp4_utils.cuh"  // vLLM device helpers (PackedVec, cvt_warp_..., SF offsets)

namespace vllm_omni {

// Apply the normalized 16-point Hadamard (H16 / 4) to the 16 values owned by
// this thread, in registers, via a Fast Walsh-Hadamard Transform. This is the
// block-diagonal Had16 the paper recommends for NVFP4 (rotation size == group
// size). FWHT costs 64 add/sub (no multiplies) instead of a dense 256-FMA
// matvec, so the rotation hides under the memory-bound quant even at large M.
//
// Natural/Sylvester order matches normalized_hadamard16 in Python. An optional
// per-lane sign vector supports a randomized Hadamard (diag(s) applied post-
// transform); the checkpoint weights must be rotated with the same signs.
// Load this thread's 16 values, apply the normalized 16-point Hadamard
// (H16 / 4) via FWHT, and leave the result in float registers -- no bf16
// write-back, so the quant epilogue below consumes floats directly. Optional
// per-lane signs implement a randomized Hadamard (weights must match).
template <class Type>
__device__ __forceinline__ void fwht16_to_float(
    vllm::PackedVec<Type, CVT_FP4_PACK16> const& vec,
    float const* __restrict__ signs, float (&x)[16]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float2 f = vllm::cast_to_float2(vec.elts[i]);
    x[2 * i] = f.x;
    x[2 * i + 1] = f.y;
  }

  // FWHT butterflies (compile-time-constant trip counts -> fully unrolled).
#pragma unroll
  for (int i = 0; i < 16; i += 2) {  // stage len=1
    float a = x[i], b = x[i + 1];
    x[i] = a + b;
    x[i + 1] = a - b;
  }
#pragma unroll
  for (int i = 0; i < 16; i += 4) {  // stage len=2
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float a = x[i + j], b = x[i + j + 2];
      x[i + j] = a + b;
      x[i + j + 2] = a - b;
    }
  }
#pragma unroll
  for (int i = 0; i < 16; i += 8) {  // stage len=4
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float a = x[i + j], b = x[i + j + 4];
      x[i + j] = a + b;
      x[i + j + 4] = a - b;
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {  // stage len=8
    float a = x[j], b = x[j + 8];
    x[j] = a + b;
    x[j + 8] = a - b;
  }

  // NOTE: these butterflies produce the *unnormalized* Hadamard, i.e. 4x the
  // normalized H16/4 result. The 1/4 normalization is folded into the quant
  // scale below (recip(24) + outputScale*0.25), which removes a 16-wide
  // multiply loop here. Random signs are +/-1, so applying them on the
  // unnormalized values is equivalent.
  if (signs != nullptr) {
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] *= signs[i];
  }
}

// NVFP4 quantization epilogue operating directly on 16 float rotated values
// (mirrors vLLM's cvt_warp_fp16_to_fp4 NVFP4 path, but float-native: no bf16
// round-trip, absmax in fp32). Writes the E4M3 scale factor and returns the
// packed E2M1 group.
// `y` holds the UNNORMALIZED Hadamard output (4x the normalized H16/4 values);
// the 1/4 normalization is folded in via recip(24) and outputScale*0.25, so the
// stored E4M3 scale and packed E2M1 codes are identical to normalizing first.
__device__ __forceinline__ vllm::u32x2 quant_fp4_from_float16(
    float (&y)[16], float SFScaleVal, uint8_t* SFout) {
  // Tree-reduction absmax (log-depth dependency chain instead of 16-serial).
  float m[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) m[i] = fmaxf(fabsf(y[2 * i]), fabsf(y[2 * i + 1]));
#pragma unroll
  for (int i = 0; i < 4; ++i) m[i] = fmaxf(m[i], m[i + 4]);
  float vecMax = fmaxf(fmaxf(m[0], m[2]), fmaxf(m[1], m[3]));

  float SFValue =
      SFScaleVal * (vecMax * vllm::reciprocal_approximate_ftz(24.0f));
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
  uint8_t fp8SFVal = reinterpret_cast<uint8_t&>(tmp);
  SFValue = float(tmp);
  if (SFout) *SFout = fp8SFVal;

  float outputScale =
      SFValue != 0.f
          ? vllm::reciprocal_approximate_ftz(
                SFValue * vllm::reciprocal_approximate_ftz(SFScaleVal)) *
                0.25f
          : 0.f;

  float2 fp2Vals[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    fp2Vals[i] = make_float2(y[2 * i] * outputScale, y[2 * i + 1] * outputScale);
  }
  return vllm::fp32_vec16_to_e2m1(fp2Vals);
}

// Each thread processes GROUPS_PER_THREAD consecutive NVFP4 groups. Handling >1
// group per thread interleaves independent FWHT + quant dependency chains (ILP),
// filling the pipeline stalls between the sequential FWHT butterfly stages,
// which helps the compute-heavy shapes (many groups per row, e.g. down_proj).
// GPT=1 is the shipping default: benchmarking on GB300 showed 2 groups/thread
// gave no meaningful gain (register pressure offset the ILP). The path is kept
// parameterizable for future tuning via -DFUSED_GROUPS_PER_THREAD=N.
#ifndef FUSED_GROUPS_PER_THREAD
#define FUSED_GROUPS_PER_THREAD 1
#endif

// Mirror of vllm::cvt_fp16_to_fp4 (swizzled-SF path) with the block-Hadamard
// folded in before the absmax/quantization epilogue.
template <class Type>
__global__ void __launch_bounds__(512)
    cvt_fp16_to_fp4_hadamard(int32_t numRows, int32_t numCols,
                             int32_t outputCols, int32_t num_padded_cols,
                             Type const* __restrict__ in,
                             float const* __restrict__ signs,
                             float const* __restrict__ SFScale,
                             uint32_t* __restrict__ out,
                             uint32_t* __restrict__ SFout) {
  using PackedVec = vllm::PackedVec<Type, CVT_FP4_PACK16>;
  constexpr int GPT = FUSED_GROUPS_PER_THREAD;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(CVT_FP4_ELTS_PER_THREAD == 16,
                "fused Hadamard path requires the PACK16 (CUDA >= 12.9) build; "
                "one thread must own a full 16-element NVFP4 group.");

  int32_t const numKTiles = (outputCols + 63) / 64;
  int sf_m = vllm::round_up<int>(numRows, 128);
  int32_t const threadCol = blockDim.x * blockIdx.y + threadIdx.x;
  int const row_groups = numCols / CVT_FP4_ELTS_PER_THREAD;

  float const global_scale = (SFScale == nullptr) ? 1.0f : SFScale[0];

  for (int rowIdx = blockIdx.x; rowIdx < sf_m; rowIdx += gridDim.x) {
    // Phase 1: load + rotate all GPT groups (independent -> compiler interleaves).
    PackedVec in_vec[GPT];
    int colIdx[GPT];
    bool valid_output[GPT];
    float y[GPT][16];
#pragma unroll
    for (int g = 0; g < GPT; ++g) {
      colIdx[g] = threadCol * GPT + g;
      int elem_idx = colIdx[g] * CVT_FP4_ELTS_PER_THREAD;
      bool in_range = colIdx[g] < num_padded_cols;
      valid_output[g] = in_range && (rowIdx < numRows) && (elem_idx < outputCols);
      // Out-of-range tail threads (num_padded_cols not a multiple of blockDim.x)
      // skip the load AND the FWHT entirely -- no dead rotation work.
      if (!in_range) continue;
      bool valid_input = (rowIdx < numRows) && (elem_idx < numCols);
      int64_t inOffset = rowIdx * row_groups + colIdx[g];
      vllm::ld256_cg_or_zero(
          reinterpret_cast<vllm::u32x8_t&>(in_vec[g]),
          &reinterpret_cast<const uint32_t*>(in)[inOffset * 8], valid_input);
      fwht16_to_float<Type>(in_vec[g], signs, y[g]);
    }

    // Phase 2: quantize + store each group.
#pragma unroll
    for (int g = 0; g < GPT; ++g) {
      if (colIdx[g] >= num_padded_cols) continue;
      auto sf_out =
          vllm::cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                                   CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx[g], numKTiles, SFout);

      // Float-native quant epilogue (no bf16 round-trip).
      auto out_val = quant_fp4_from_float16(y[g], global_scale, sf_out);

      if (valid_output[g]) {
        int64_t outOffset = rowIdx * (outputCols / 8) + colIdx[g] * 2;
        uint64_t packed64 = (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
        reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
      }
    }
  }
}

}  // namespace vllm_omni

// Host entry: quantize `input` [m, n] (fp16/bf16) to NVFP4 with a fused block-16
// Hadamard (FWHT), returning (packed_fp4 [m, n/2] uint8, swizzled scales
// fp8_e4m3fn [round_up(m,128), round_up(n/16,4)]) -- byte-identical layout to
// vLLM scaled_fp4_quant(is_sf_swizzled_layout=True), a drop-in for
// cutlass_scaled_fp4_mm. `signs` is an optional [16] float32 (+/-1) for a
// randomized Hadamard (nullptr = plain Had16). `global_scale` is a scalar
// float32 tensor (the next GEMM's alpha component).
std::tuple<torch::Tensor, torch::Tensor> fused_hadamard_nvfp4_quant(
    torch::Tensor const& input, torch::Tensor const& global_scale,
    std::optional<torch::Tensor> const& signs) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  const at::cuda::CUDAGuard device_guard(input.device());
  TORCH_CHECK(input.dim() == 2, "input must be 2D [m, n]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16 ||
                  input.scalar_type() == torch::kHalf,
              "input must be fp16 or bf16");
  TORCH_CHECK(global_scale.is_cuda() &&
                  global_scale.scalar_type() == torch::kFloat32,
              "global_scale must be a float32 CUDA scalar tensor");

  int32_t m = static_cast<int32_t>(input.size(0));
  int32_t n = static_cast<int32_t>(input.size(1));
  TORCH_CHECK(n % 16 == 0, "n must be a multiple of 16");
  int32_t output_n = n;  // no weight-padding in this path

  auto output = torch::empty({m, n / 2}, input.options().dtype(torch::kUInt8));

  // Swizzled SF: (round_up(m,128), round_up(n/16,4)) fp8_e4m3fn. The GEMM reads
  // padded scale factors the kernel may not write, so those must be zero -- but
  // when m and n/16 are already aligned there is NO padding and the kernel
  // writes every element, so we can skip the (serialized) zeroing memset.
  int64_t rounded_m = vllm::round_up<int64_t>(m, 128);
  int64_t scale_n = n / CVT_FP4_SF_VEC_SIZE;
  int64_t rounded_n = vllm::round_up<int64_t>(scale_n, 4);
  bool sf_no_padding = (rounded_m == m) && (rounded_n == scale_n);
  auto sf_opts = input.options().dtype(torch::kFloat8_e4m3fn);
  auto output_sf = sf_no_padding
                       ? torch::empty({rounded_m, rounded_n}, sf_opts)
                       : torch::zeros({rounded_m, rounded_n}, sf_opts);

  float const* signs_ptr = nullptr;
  torch::Tensor signs_c;
  if (signs.has_value()) {
    signs_c = signs->contiguous();
    TORCH_CHECK(signs_c.is_cuda() &&
                    signs_c.scalar_type() == torch::kFloat32 &&
                    signs_c.numel() == 16,
                "signs must be a [16] float32 CUDA tensor");
    signs_ptr = static_cast<float const*>(signs_c.data_ptr());
  }
  float const* sf_scale_ptr = static_cast<float const*>(global_scale.data_ptr());

  // SM count is device-invariant; query once per device (host-call overhead
  // otherwise recurs on every forward).
  int dev = input.get_device();
  static int mp_by_dev[16] = {0};
  int mp_count;
  if (dev >= 0 && dev < 16) {
    if (mp_by_dev[dev] == 0) {
      cudaDeviceGetAttribute(&mp_by_dev[dev], cudaDevAttrMultiProcessorCount, dev);
    }
    mp_count = mp_by_dev[dev];
  } else {
    cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, dev);
  }

  int32_t output_sf_n_unpadded = output_n / CVT_FP4_SF_VEC_SIZE;
  int sf_n_int = int(vllm::round_up(output_sf_n_unpadded, 4) / 4);
  int32_t num_padded_cols =
      sf_n_int * 4 * CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  // Each thread handles FUSED_GROUPS_PER_THREAD consecutive group-columns.
  // Balance threads across grid_y so blockDim.x * grid_y hugs threads_needed
  // (e.g. 768 -> block=384, grid_y=2, zero idle threads) instead of
  // min(.,512) which would launch 1024 threads for 768 groups.
  int threads_needed =
      vllm::div_round_up(num_padded_cols, FUSED_GROUPS_PER_THREAD);
  int grid_y = vllm::div_round_up(threads_needed, 512);
  dim3 block(vllm::div_round_up(threads_needed, grid_y));
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  AT_DISPATCH_SWITCH(
      input.scalar_type(), "fused_hadamard_nvfp4_quant",
      AT_DISPATCH_CASE(torch::kBFloat16,
                       [&] {
                         auto kernel =
                             vllm_omni::cvt_fp16_to_fp4_hadamard<__nv_bfloat16>;
                         // Occupancy depends only on the kernel + block size;
                         // memoize to avoid a driver query on every forward.
                         static int memo_block = -1, memo_bps = 1;
                         if (memo_block != static_cast<int>(block.x)) {
                           cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &memo_bps, kernel, static_cast<int>(block.x), 0);
                           memo_block = static_cast<int>(block.x);
                         }
                         int grid_x = std::min(
                             static_cast<int>(rounded_m),
                             std::max(1, mp_count * std::max(1, memo_bps) /
                                             std::max(1, grid_y)));
                         dim3 grid(grid_x, grid_y);
                         kernel<<<grid, block, 0, stream>>>(
                             m, n, output_n, num_padded_cols,
                             static_cast<__nv_bfloat16 const*>(input.data_ptr()),
                             signs_ptr, sf_scale_ptr,
                             reinterpret_cast<uint32_t*>(output.data_ptr()),
                             reinterpret_cast<uint32_t*>(output_sf.data_ptr()));
                       })
          AT_DISPATCH_CASE(torch::kHalf, [&] {
            auto kernel = vllm_omni::cvt_fp16_to_fp4_hadamard<half>;
            static int memo_block = -1, memo_bps = 1;
            if (memo_block != static_cast<int>(block.x)) {
              cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &memo_bps, kernel, static_cast<int>(block.x), 0);
              memo_block = static_cast<int>(block.x);
            }
            int grid_x = std::min(
                static_cast<int>(rounded_m),
                std::max(1, mp_count * std::max(1, memo_bps) /
                                std::max(1, grid_y)));
            dim3 grid(grid_x, grid_y);
            kernel<<<grid, block, 0, stream>>>(
                m, n, output_n, num_padded_cols,
                static_cast<half const*>(input.data_ptr()), signs_ptr,
                sf_scale_ptr,
                reinterpret_cast<uint32_t*>(output.data_ptr()),
                reinterpret_cast<uint32_t*>(output_sf.data_ptr()));
          }));

  return {output, output_sf};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_hadamard_nvfp4_quant", &fused_hadamard_nvfp4_quant,
        "Fused block-16 Hadamard (FWHT) + NVFP4 quantization (swizzled SF)",
        pybind11::arg("input"), pybind11::arg("global_scale"),
        pybind11::arg("signs") = std::nullopt);
}
