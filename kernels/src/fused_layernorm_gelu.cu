#include "kernels/fused_layernorm_gelu.h"
#include "common.cuh"

namespace ck {

// LayerNorm + GELU collapsed into one kernel: a single read of x and a single
// write of y, with the layernorm statistics reused across the two logical ops.
__global__ void layernorm_gelu_kernel(const float* __restrict__ in,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      float* __restrict__ out, int rows,
                                      int cols, float eps) {
  extern __shared__ float smem[];
  const int row = blockIdx.x;
  const float* x = in + (size_t)row * cols;
  float* y = out + (size_t)row * cols;

  float sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) sum += x[i];
  const float mean = block_reduce_sum(sum, smem) / (float)cols;

  float sq = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float d = x[i] - mean;
    sq += d * d;
  }
  const float var = block_reduce_sum(sq, smem) / (float)cols;
  const float rstd = rsqrtf(var + eps);

  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    y[i] = gelu_op((x[i] - mean) * rstd * gamma[i] + beta[i]);
}

void fused_layernorm_gelu_forward(const float* in, const float* gamma,
                                  const float* beta, float* out, int rows,
                                  int cols, float eps, cudaStream_t stream) {
  const int threads = 256;
  const int smem_bytes = 32 * sizeof(float);
  layernorm_gelu_kernel<<<rows, threads, smem_bytes, stream>>>(
      in, gamma, beta, out, rows, cols, eps);
}

}  // namespace ck
