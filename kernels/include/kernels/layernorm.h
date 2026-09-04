#pragma once

// Layer Normalization over the last dimension of a 2-D row-major tensor
// [rows, cols], with per-column learnable scale `gamma` / shift `beta`:
//   mean[r]   = mean_c in[r][c]
//   var[r]    = mean_c (in[r][c] - mean[r])^2
//   out[r][c] = (in[r][c] - mean[r]) / sqrt(var[r] + eps) * gamma[c] + beta[c]
//
// One block per row; a two-pass reduction computes the mean, then the
// variance, then the affine transform is applied in a single final pass.

#include <cuda_runtime.h>

namespace ck {

// `gamma` / `beta` are device pointers of length `cols`.
void layernorm_forward(const float* in, const float* gamma, const float* beta,
                       float* out, int rows, int cols, float eps = 1e-5f,
                       cudaStream_t stream = 0);

}  // namespace ck
