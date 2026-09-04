#include "kernels/rope.h"
#include "test_utils.hpp"

#include <cmath>

static std::vector<float> cpu_rope(const std::vector<float>& x, int T, int D,
                                   float base, int start_pos) {
  std::vector<float> y(x.size());
  const int n_pairs = D / 2;
  for (int p = 0; p < T; ++p)
    for (int i = 0; i < n_pairs; ++i) {
      const float theta =
          (float)(start_pos + p) / std::pow(base, (2.0f * i) / (float)D);
      const float c = std::cos(theta), s = std::sin(theta);
      const int off = p * D;
      const float x0 = x[off + 2 * i], x1 = x[off + 2 * i + 1];
      y[off + 2 * i] = x0 * c - x1 * s;
      y[off + 2 * i + 1] = x0 * s + x1 * c;
    }
  return y;
}

void test_rope() {
  const int T = 8, D = 64;
  const float base = 10000.0f;
  const int start_pos = 5;
  const auto x = ktest::randn(T * D, 0.0f, 1.0f);

  ktest::DeviceArray<float> dx(T * D), dy(T * D);
  dx.upload(x);
  ck::rope_forward(dx.ptr, dy.ptr, T, D, base, start_pos);
  const auto got = dy.download();

  const auto ref = cpu_rope(x, T, D, base, start_pos);
  ktest::close("rope", got, ref, 1e-4f, 1e-5f);
}
