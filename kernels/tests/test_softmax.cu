#include "kernels/softmax.h"
#include "test_utils.hpp"

#include <cmath>

static std::vector<float> cpu_softmax(const std::vector<float>& x, int rows,
                                      int cols) {
  std::vector<float> y(x.size());
  for (int r = 0; r < rows; ++r) {
    const float* row = &x[(size_t)r * cols];
    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = std::fmaxf(m, row[c]);
    float s = 0.0f;
    for (int c = 0; c < cols; ++c) s += std::exp(row[c] - m);
    for (int c = 0; c < cols; ++c)
      y[(size_t)r * cols + c] = std::exp(row[c] - m) / s;
  }
  return y;
}

void test_softmax() {
  const int rows = 8, cols = 256;
  const auto x = ktest::randu(rows * cols, -5.0f, 5.0f);

  ktest::DeviceArray<float> dx(rows * cols), dy(rows * cols);
  dx.upload(x);
  ck::softmax_forward(dx.ptr, dy.ptr, rows, cols);
  const auto got = dy.download();

  const auto ref = cpu_softmax(x, rows, cols);
  ktest::close("softmax", got, ref, 1e-3f, 1e-4f);
}
