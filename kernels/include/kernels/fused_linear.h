#pragma once

// Fused Linear layer: GEMM + bias + GELU in a single kernel.
//
//   C[M, N] = GELU( A[M, K] @ B[K, N] + bias[N] )
//
// The tiled GEMM keeps its accumulator in registers across the whole K loop,
// and the bias + GELU are applied only once in the epilogue, right before the
// single write to C. This is the same epilogue-fusion pattern TensorRT /
// CUTLASS use to eliminate a full GEMM output round trip + two extra passes.

#include <cuda_runtime.h>

namespace ck {

// A is [M, K], B is [K, N], C is [M, N], bias is [N] (all device pointers).
void fused_linear_gelu_forward(const float* A, const float* B, const float* bias,
                               float* C, int M, int N, int K,
                               cudaStream_t stream = 0);

}  // namespace ck
