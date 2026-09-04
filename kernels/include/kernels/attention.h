#pragma once

// Single-head scaled dot-product attention with online softmax, fused into a
// single kernel (one thread block per query):
//
//   S[q, j] = (Q[q] . K[j]) * scale            (scale = 1 / sqrt(D))
//   P[q, :] = softmax(S[q, :])
//   O[q, :] = sum_j P[q, j] * V[j, :]
//
// Instead of materializing the [T, T] score matrix, each block sweeps the keys
// once and keeps a running max / sum to compute a numerically stable softmax
// (the "online softmax" trick behind FlashAttention). Requires D == blockDim.x.

#include <cuda_runtime.h>

namespace ck {

// Q / K / V / O are [T, D] row-major device tensors. `scale` is typically
// 1 / sqrt(D) and is passed in explicitly so callers control the scaling.
void attention_forward(const float* Q, const float* K, const float* V, float* O,
                       int T, int D, float scale, cudaStream_t stream = 0);

}  // namespace ck
