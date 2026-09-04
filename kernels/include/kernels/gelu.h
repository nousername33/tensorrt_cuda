#pragma once

// Elementwise GELU activation (tanh approximation):
//   out[i] = 0.5 * in[i] * (1 + tanh(sqrt(2/pi) * (in[i] + 0.044715 * in[i]^3)))

#include <cuda_runtime.h>

namespace ck {

// `in` / `out` are device pointers to arrays of `n` floats.
void gelu_forward(const float* in, float* out, int n, cudaStream_t stream = 0);

}  // namespace ck
