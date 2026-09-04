#include "kernels/rope.h"
#include <cmath>

namespace ck {

// Each thread rotates one (position, pair-of-dims) element.
__global__ void rope_kernel(const float* __restrict__ in,
                            float* __restrict__ out, int T, int D, float base,
                            int start_pos) {
  const int n_pairs = D / 2;
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= T * n_pairs) return;

  const int p = pair / n_pairs;   // token position
  const int i = pair % n_pairs;   // dimension pair index

  const float theta =
      (float)(start_pos + p) / powf(base, (2.0f * i) / (float)D);
  const float c = cosf(theta);
  const float s = sinf(theta);

  const int off = p * D;
  const float x0 = in[off + 2 * i];
  const float x1 = in[off + 2 * i + 1];
  out[off + 2 * i] = x0 * c - x1 * s;
  out[off + 2 * i + 1] = x0 * s + x1 * c;
}

void rope_forward(const float* in, float* out, int T, int D, float base,
                  int start_pos, cudaStream_t stream) {
  const int n_pairs = D / 2;
  const int total = T * n_pairs;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  rope_kernel<<<blocks, threads, 0, stream>>>(in, out, T, D, base, start_pos);
}

}  // namespace ck
