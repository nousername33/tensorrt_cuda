#include "kernels/rmsnorm.h"
#include "common.cuh"

namespace ck {

__global__ void rmsnorm_kernel(const float* __restrict__ in,
                               const float* __restrict__ gamma,
                               float* __restrict__ out, int rows, int cols,
                               float eps) {
  extern __shared__ float smem[];  // >= 32 floats for block reduction
  const int row = blockIdx.x;
  const float* x = in + (size_t)row * cols;
  float* y = out + (size_t)row * cols;

  // Single reduction pass: mean of squares.
  float sq = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) sq += x[i] * x[i];
  const float ms = block_reduce_sum(sq, smem) / (float)cols;
  const float rstd = rsqrtf(ms + eps);

  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    y[i] = x[i] * rstd * gamma[i];
}

void rmsnorm_forward(const float* in, const float* gamma, float* out, int rows,
                     int cols, float eps, cudaStream_t stream) {
  const int threads = 256;
  const int smem_bytes = 32 * sizeof(float);
  rmsnorm_kernel<<<rows, threads, smem_bytes, stream>>>(in, gamma, out, rows,
                                                        cols, eps);
}

}  // namespace ck
