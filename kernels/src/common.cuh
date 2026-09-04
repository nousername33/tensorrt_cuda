#pragma once

// Shared device-side helpers used across kernels:
//  - warp / block reductions (sum, max) with the result broadcast to all threads
//  - GELU activation
//
// These live in src/ (not the public include dir) because they are only needed
// by the .cu implementation files, not by host code.

#include <cuda_runtime.h>

namespace ck {

// ---------------------------------------------------------------------------
// Warp-level reductions (assume full 32-lane warp participation)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
  }
  return v;
}

// ---------------------------------------------------------------------------
// Block-level reductions.
//
// `smem` must point to at least 32 floats of shared memory (enough to hold one
// partial per warp, since a block has at most 1024 threads = 32 warps).
// The final result is written back to smem[0] and broadcast to every thread.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  const int nwarps = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
  if (wid == 0) {
    v = warp_reduce_sum(v);
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;

  v = warp_reduce_max(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  const int nwarps = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : -INFINITY;
  if (wid == 0) {
    v = warp_reduce_max(v);
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

// ---------------------------------------------------------------------------
// GELU (tanh approximation), as used in BERT / GPT-style models.
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ---------------------------------------------------------------------------
__device__ __forceinline__ float gelu_op(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2 / pi)
  const float k = 0.044715f;
  const float t = tanhf(c * (x + k * x * x * x));
  return 0.5f * x * (1.0f + t);
}

}  // namespace ck
