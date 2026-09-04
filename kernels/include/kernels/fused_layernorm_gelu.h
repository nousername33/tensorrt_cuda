#pragma once

// Fused LayerNorm -> GELU.
//
// Computes, in a single kernel and without an intermediate global-memory
// round trip:
//   out[r][c] = GELU( (in[r][c] - mean[r]) / sqrt(var[r] + eps) * gamma[c] + beta[c] )
//
// This is a classic operator-fusion example: two logical ops (layernorm +
// gelu) collapse into one read + one write per element.

#include <cuda_runtime.h>

namespace ck {

void fused_layernorm_gelu_forward(const float* in, const float* gamma,
                                  const float* beta, float* out, int rows,
                                  int cols, float eps = 1e-5f,
                                  cudaStream_t stream = 0);

}  // namespace ck
