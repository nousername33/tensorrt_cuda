#include "kernels/fused_bias_gelu.h"
#include "common.cuh"

namespace ck {

__global__ void bias_gelu_kernel(const float* __restrict__ in,
                                 const float* __restrict__ bias,
                                 float* __restrict__ out, int rows, int cols) {
  const int total = rows * cols;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int col = i % cols;
  out[i] = gelu_op(in[i] + bias[col]);
}

void fused_bias_gelu_forward(const float* in, const float* bias, float* out,
                             int rows, int cols, cudaStream_t stream) {
  const int total = rows * cols;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  bias_gelu_kernel<<<blocks, threads, 0, stream>>>(in, bias, out, rows, cols);
}

}  // namespace ck
