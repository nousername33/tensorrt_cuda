#include "kernels/attention.h"
#include "common.cuh"

namespace ck {

// One block per query row. blockDim.x == D (each thread owns one head-dim
// element of the output accumulator, plus one element of Q/K/V rows).
//
// Online softmax: we never materialize the [T, T] scores. Instead we keep a
// running max m and sum l and rescale the running output accumulator each time
// the max grows. Numerically identical to the two-pass stable softmax.
__global__ void attention_kernel(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 float* __restrict__ O, int T, int D,
                                 float scale) {
  extern __shared__ float smem[];  // >= 32 floats for the block-reduced dot
  const int q = blockIdx.x;
  const int tid = threadIdx.x;

  const float qi = Q[(size_t)q * D + tid];
  float m = -INFINITY;  // running max of scores
  float l = 0.0f;       // running sum of exp(scores - m)
  float o = 0.0f;       // running accumulator for O[q, tid]

  for (int j = 0; j < T; ++j) {
    // score = (Q[q] . K[j]) * scale, reduced across the block.
    const float kj = K[(size_t)j * D + tid];
    const float s = block_reduce_sum(qi * kj, smem) * scale;

    const float m_new = fmaxf(m, s);
    const float p = __expf(s - m_new);
    const float alpha = __expf(m - m_new);  // rescale factor for prior acc

    l = l * alpha + p;
    o = o * alpha + p * V[(size_t)j * D + tid];
    m = m_new;
  }

  O[(size_t)q * D + tid] = o / l;
}

void attention_forward(const float* Q, const float* K, const float* V, float* O,
                       int T, int D, float scale, cudaStream_t stream) {
  // The kernel assumes one thread per head dimension.
  const int smem_bytes = 32 * sizeof(float);
  attention_kernel<<<T, D, smem_bytes, stream>>>(Q, K, V, O, T, D, scale);
}

}  // namespace ck
