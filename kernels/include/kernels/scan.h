#pragma once

// Inclusive prefix sum (scan) of `n` floats, using the work-efficient
// Blelloch up-sweep / down-sweep algorithm in a single thread block.
//
//   out[i] = in[0] + in[1] + ... + in[i]
//
// Scan is a fundamental parallel primitive (log-depth, work-efficient) that
// underlies stream compaction, radix sort, and segmented reductions.

#include <cuda_runtime.h>

namespace ck {

// Requirements: `n` is a power of two and equals blockDim.x (currently a
// single-block implementation; 512 or 1024 are good test sizes).
void scan_inclusive_forward(const float* in, float* out, int n,
                            cudaStream_t stream = 0);

}  // namespace ck
