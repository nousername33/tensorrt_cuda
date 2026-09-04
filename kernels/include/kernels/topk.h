#pragma once

// Top-K selection: find the `k` largest values in `in` (length `n`) and return
// both the values and their original indices (indices are stable w.r.t. the
// input order for ties).
//
// Algorithm (single block): every thread scans a strided slice of the input
// maintaining a private, insertion-sorted top-k list; the per-thread lists are
// then merged pairwise in shared memory (a log(blockDim.x)-depth tournament)
// until thread 0 holds the global top-k.

#include <cuda_runtime.h>

namespace ck {

// `out_vals` / `out_idx` are device buffers of length `k`. Requires k <= 32.
void topk_forward(const float* in, int n, float* out_vals, int* out_idx, int k,
                  cudaStream_t stream = 0);

}  // namespace ck
