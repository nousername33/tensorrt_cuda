#pragma once

// RMSNorm (Root Mean Square Layer Normalization), used in LLaMA / Mistral
// style models. Unlike layernorm it drops the mean subtraction and bias:
//   rms[r]     = sqrt(mean_c in[r][c]^2 + eps)
//   out[r][c]  = in[r][c] / rms[r] * gamma[c]
//
// Cheaper than layernorm (a single reduction pass) while keeping the input
// scale-invariant.

#include <cuda_runtime.h>

namespace ck {

// `gamma` is a device pointer of length `cols`.
void rmsnorm_forward(const float* in, const float* gamma, float* out,
                     int rows, int cols, float eps = 1e-5f,
                     cudaStream_t stream = 0);

}  // namespace ck
