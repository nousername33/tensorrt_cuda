#pragma once

// Tiled dense matrix multiplication (row-major, no transpose):
//   C[M, N] = A[M, K] @ B[K, N]
//
// A/B tiles are staged in shared memory to amortize global loads; each thread
// computes one output element, accumulating over the K dimension in blocks of
// TILE (=32).

#include <cuda_runtime.h>

namespace ck {

// A is [M, K], B is [K, N], C is [M, N] (all device pointers, row-major).
void matmul_forward(const float* A, const float* B, float* C, int M, int N,
                    int K, cudaStream_t stream = 0);

}  // namespace ck
