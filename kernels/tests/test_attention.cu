#include "kernels/attention.h"
#include "test_utils.hpp"

#include <cmath>

static std::vector<float> cpu_attention(const std::vector<float>& Q,
                                        const std::vector<float>& K,
                                        const std::vector<float>& V, int T,
                                        int D, float scale) {
  std::vector<float> O((size_t)T * D, 0.0f);
  for (int q = 0; q < T; ++q) {
    std::vector<float> s(T, 0.0f);
    float m = -INFINITY;
    for (int j = 0; j < T; ++j) {
      float dot = 0.0f;
      for (int d = 0; d < D; ++d)
        dot += Q[(size_t)q * D + d] * K[(size_t)j * D + d];
      s[j] = dot * scale;
      m = std::fmaxf(m, s[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < T; ++j) {
      s[j] = std::exp(s[j] - m);
      sum += s[j];
    }
    for (int j = 0; j < T; ++j)
      for (int d = 0; d < D; ++d)
        O[(size_t)q * D + d] += (s[j] / sum) * V[(size_t)j * D + d];
  }
  return O;
}

void test_attention() {
  const int T = 16, D = 64;
  const float scale = 1.0f / std::sqrt((float)D);
  // Small values keep the scores well-conditioned for a tight tolerance.
  const auto Q = ktest::randn(T * D, 0.0f, 0.1f);
  const auto K = ktest::randn(T * D, 0.0f, 0.1f);
  const auto V = ktest::randn(T * D, 0.0f, 0.1f);

  ktest::DeviceArray<float> dQ(T * D), dK(T * D), dV(T * D), dO(T * D);
  dQ.upload(Q);
  dK.upload(K);
  dV.upload(V);
  ck::attention_forward(dQ.ptr, dK.ptr, dV.ptr, dO.ptr, T, D, scale);
  const auto got = dO.download();

  const auto ref = cpu_attention(Q, K, V, T, D, scale);
  ktest::close("attention", got, ref, 1e-2f, 1e-3f);
}
