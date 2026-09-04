#include "kernels/topk.h"

namespace ck {

#define MAXK 32

// Insert (v, idx) into a descending-sorted list of length `cnt` (<= k).
// Elements are shifted right to make room; when the list is full the last
// (smallest) element is dropped. `pos` is clamped to k-1 so the shift never
// writes past the array when cnt == k.
__device__ __forceinline__ void insert_topk(float v, int idx, float* vals,
                                            int* idxs, int& cnt, int k) {
  if (cnt == k && v <= vals[k - 1]) return;  // not in the current top-k
  int pos = (cnt < k) ? cnt : (k - 1);
  while (pos > 0 && vals[pos - 1] < v) {
    vals[pos] = vals[pos - 1];
    idxs[pos] = idxs[pos - 1];
    --pos;
  }
  vals[pos] = v;
  idxs[pos] = idx;
  if (cnt < k) ++cnt;
}

// Merge two descending length-k lists into `out` (descending, length k).
__device__ __forceinline__ void merge_topk(const float* a_vals,
                                           const int* a_idx, const float* b_vals,
                                           const int* b_idx, float* out_vals,
                                           int* out_idx, int k) {
  int i = 0, j = 0, o = 0;
  while (o < k) {
    if (j >= k || (i < k && a_vals[i] >= b_vals[j])) {
      out_vals[o] = a_vals[i];
      out_idx[o] = a_idx[i];
      ++i;
    } else {
      out_vals[o] = b_vals[j];
      out_idx[o] = b_idx[j];
      ++j;
    }
    ++o;
  }
}

__global__ void topk_kernel(const float* __restrict__ in, int n,
                            float* __restrict__ out_vals,
                            int* __restrict__ out_idx, int k) {
  extern __shared__ char raw[];
  float* smem_vals = reinterpret_cast<float*>(raw);              // [blockDim * k]
  int* smem_idx = reinterpret_cast<int*>(raw + (size_t)blockDim.x * k * sizeof(float));

  const int tid = threadIdx.x;

  // 1) Each thread keeps a private top-k of its strided slice.
  float lv[MAXK];
  int li[MAXK];
  int cnt = 0;
  for (int i = tid; i < n; i += blockDim.x) insert_topk(in[i], i, lv, li, cnt, k);

  // 2) Pad to length k and publish to shared memory.
  for (int j = cnt; j < k; ++j) {
    lv[j] = -INFINITY;
    li[j] = -1;
  }
  for (int j = 0; j < k; ++j) {
    smem_vals[tid * k + j] = lv[j];
    smem_idx[tid * k + j] = li[j];
  }
  __syncthreads();

  // 3) Pairwise merge tournament over the per-thread lists.
  int width = blockDim.x;
  while (width > 1) {
    const int half = width >> 1;
    if (tid < half) {
      float tv[MAXK];
      int ti[MAXK];
      merge_topk(smem_vals + tid * k, smem_idx + tid * k,
                 smem_vals + (tid + half) * k, smem_idx + (tid + half) * k, tv,
                 ti, k);
      for (int j = 0; j < k; ++j) {
        smem_vals[tid * k + j] = tv[j];
        smem_idx[tid * k + j] = ti[j];
      }
    }
    __syncthreads();
    width = half;
  }

  // 4) Thread 0 copies the merged global top-k to the output.
  if (tid == 0) {
    for (int j = 0; j < k; ++j) {
      out_vals[j] = smem_vals[j];
      out_idx[j] = smem_idx[j];
    }
  }
}

void topk_forward(const float* in, int n, float* out_vals, int* out_idx, int k,
                  cudaStream_t stream) {
  const int threads = 256;
  const int smem_bytes = threads * k * (int)(sizeof(float) + sizeof(int));
  topk_kernel<<<1, threads, smem_bytes, stream>>>(in, n, out_vals, out_idx, k);
}

}  // namespace ck
