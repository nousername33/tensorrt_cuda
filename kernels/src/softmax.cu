#include "kernels/softmax.h"
#include "common.cuh"

namespace ck {

__global__ void softmax_row_kernel(const float* __restrict__ in,
                                   float* __restrict__ out, int rows, int cols) {
  extern __shared__ float smem[];  // >= 32 floats, used for block reduction
  const int row = blockIdx.x;
  const float* x = in + (size_t)row * cols;
  float* y = out + (size_t)row * cols;

  // Pass 1: row max.
  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    local_max = fmaxf(local_max, x[i]);
  const float row_max = block_reduce_max(local_max, smem);

  // Pass 2: sum of exp(x - max).
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    local_sum += __expf(x[i] - row_max);
  const float row_sum = block_reduce_sum(local_sum, smem);

  // Pass 3: normalize.
  const float inv = 1.0f / row_sum;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    y[i] = __expf(x[i] - row_max) * inv;
}

void softmax_forward(const float* in, float* out, int rows, int cols,
                     cudaStream_t stream) {
  const int threads = 256;
  const int smem_bytes = 32 * sizeof(float);
  softmax_row_kernel<<<rows, threads, smem_bytes, stream>>>(in, out, rows, cols);
}

}  // namespace ck
