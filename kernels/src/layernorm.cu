#include "kernels/layernorm.h"
#include "common.cuh"

namespace ck {

__global__ void layernorm_kernel(const float* __restrict__ in,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ out, int rows, int cols,
                                 float eps) {
  extern __shared__ float smem[];  // >= 32 floats for block reduction
  const int row = blockIdx.x;
  const float* x = in + (size_t)row * cols;
  float* y = out + (size_t)row * cols;

  // Pass 1: mean.
  float sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) sum += x[i];
  const float mean = block_reduce_sum(sum, smem) / (float)cols;

  // Pass 2: variance.
  float sq = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float d = x[i] - mean;
    sq += d * d;
  }
  const float var = block_reduce_sum(sq, smem) / (float)cols;
  const float rstd = rsqrtf(var + eps);

  // Pass 3: normalize + affine.
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    y[i] = (x[i] - mean) * rstd * gamma[i] + beta[i];
}

void layernorm_forward(const float* in, const float* gamma, const float* beta,
                       float* out, int rows, int cols, float eps,
                       cudaStream_t stream) {
  const int threads = 256;
  const int smem_bytes = 32 * sizeof(float);
  layernorm_kernel<<<rows, threads, smem_bytes, stream>>>(in, gamma, beta, out,
                                                          rows, cols, eps);
}

}  // namespace ck
