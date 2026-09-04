#include "kernels/matmul.h"

namespace ck {

#define TILE 32

// C[M, N] = A[M, K] @ B[K, N], row-major. One thread computes one C element.
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;  // output row
  const int col = blockIdx.x * TILE + tx;  // output col

  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += TILE) {
    // Cooperative load of one A tile and one B tile into shared memory.
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

  if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

void matmul_forward(const float* A, const float* B, float* C, int M, int N,
                    int K, cudaStream_t stream) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

}  // namespace ck
