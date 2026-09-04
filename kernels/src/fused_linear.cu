#include "kernels/fused_linear.h"
#include "common.cuh"

namespace ck {

#define TILE 32

// Tiled GEMM with a fused bias + GELU epilogue. The accumulator stays in a
// register for the whole K loop; bias/GELU touch it only once, before the
// single write to C.
__global__ void linear_gelu_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   const float* __restrict__ bias,
                                   float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += TILE) {
    if (row < M && (k0 + tx) < K)
      As[ty][tx] = A[(size_t)row * K + k0 + tx];
    else
      As[ty][tx] = 0.0f;

    if ((k0 + ty) < K && col < N)
      Bs[ty][tx] = B[(size_t)(k0 + ty) * N + col];
    else
      Bs[ty][tx] = 0.0f;

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];

    __syncthreads();
  }

  if (row < M && col < N) C[(size_t)row * N + col] = gelu_op(acc + bias[col]);
}

void fused_linear_gelu_forward(const float* A, const float* B, const float* bias,
                               float* C, int M, int N, int K,
                               cudaStream_t stream) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  linear_gelu_kernel<<<grid, block, 0, stream>>>(A, B, bias, C, M, N, K);
}

}  // namespace ck
