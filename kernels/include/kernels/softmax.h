#pragma once

// Numerically stable softmax along the last dimension of a 2-D row-major
// tensor of shape [rows, cols]. Each row is normalized independently:
//   out[r][c] = exp(in[r][c] - max_c in[r][c]) / sum_c exp(...)
//
// One thread block is launched per row; the block computes the row max and the
// sum of exponentials using a two-pass block reduction (see common.cuh).

#include <cuda_runtime.h>

namespace ck {

// `in` / `out` are device pointers to arrays of `rows * cols` floats.
void softmax_forward(const float* in, float* out, int rows, int cols,
                     cudaStream_t stream = 0);

}  // namespace ck
