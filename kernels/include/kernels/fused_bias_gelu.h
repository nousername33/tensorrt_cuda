#pragma once

// Fused bias-add -> GELU.
//
// Computes, in one elementwise kernel:
//   out[r][c] = GELU( in[r][c] + bias[c] )
//
// where `bias` (length `cols`) is broadcast across rows. Fusion avoids
// materializing the intermediate `in + bias` tensor to global memory.

#include <cuda_runtime.h>

namespace ck {

// `in` / `out` are device pointers of `rows * cols` floats; `bias` has `cols`.
void fused_bias_gelu_forward(const float* in, const float* bias, float* out,
                             int rows, int cols, cudaStream_t stream = 0);

}  // namespace ck
