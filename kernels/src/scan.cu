#include "kernels/scan.h"

namespace ck {

// Work-efficient Blelloch inclusive scan in a single block. Requires
// n == blockDim.x and n a power of two (enforced by the launcher).
//
// Up-sweep builds a reduction tree in shared memory; down-sweep redistributes
// partial sums, producing the *exclusive* scan. Adding the original element
// back yields the inclusive scan.
__global__ void scan_inclusive_kernel(const float* __restrict__ in,
                                      float* __restrict__ out, int n) {
  extern __shared__ float s[];
  const int tid = threadIdx.x;

  s[tid] = in[tid];
  __syncthreads();

  // Up-sweep (reduce).
  for (int stride = 2; stride <= n; stride <<= 1) {
    const int k = (tid + 1) * stride - 1;
    if (k < n) s[k] += s[k - (stride >> 1)];
    __syncthreads();
  }

  // Clear the last element to the identity (exclusive scan).
  if (tid == 0) s[n - 1] = 0.0f;
  __syncthreads();

  // Down-sweep.
  for (int stride = n; stride >= 2; stride >>= 1) {
    const int k = (tid + 1) * stride - 1;
    if (k < n) {
      const float t = s[k];
      s[k] += s[k - (stride >> 1)];
      s[k - (stride >> 1)] = t;
    }
    __syncthreads();
  }

  out[tid] = in[tid] + s[tid];  // exclusive + original == inclusive
}

void scan_inclusive_forward(const float* in, float* out, int n,
                            cudaStream_t stream) {
  const int threads = n;  // n is a power of two and equals blockDim.x
  const int smem_bytes = n * sizeof(float);
  scan_inclusive_kernel<<<1, threads, smem_bytes, stream>>>(in, out, n);
}

}  // namespace ck
