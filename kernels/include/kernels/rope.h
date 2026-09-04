#pragma once

// Rotary Position Embeddings (RoPE), used in LLaMA / GPT-NeoX / PaLM.
//
// For a token at position p and head-dimension pair (2i, 2i+1):
//   theta          = p / base^(2i / D)
//   out[p, 2i]     =  in[p, 2i] * cos(theta) - in[p, 2i+1] * sin(theta)
//   out[p, 2i+1]   =  in[p, 2i] * sin(theta) + in[p, 2i+1] * cos(theta)
//
// Each thread rotates one (position, pair) element; no cross-thread
// communication is needed, but the angle computation is per-position.

#include <cuda_runtime.h>

namespace ck {

// `in` / `out` are [T, D] row-major device tensors; D must be even.
// `base` is the rotary base (e.g. 10000.0); `start_pos` is the absolute
// position of the first token (0 for a standalone sequence).
void rope_forward(const float* in, float* out, int T, int D, float base = 10000.0f,
                  int start_pos = 0, cudaStream_t stream = 0);

}  // namespace ck
