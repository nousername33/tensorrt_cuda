#include "kernels/gelu.h"
#include "common.cuh"

namespace ck {

__global__ void gelu_kernel(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = gelu_op(in[i]);
}

void gelu_forward(const float* in, float* out, int n, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  gelu_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

}  // namespace ck
