#include "kernels/rmsnorm.h"
#include "test_utils.hpp"

#include <cmath>

static std::vector<float> cpu_rmsnorm(const std::vector<float>& x,
                                      const std::vector<float>& gamma, int rows,
                                      int cols, float eps) {
  std::vector<float> y(x.size());
  for (int r = 0; r < rows; ++r) {
    const float* row = &x[(size_t)r * cols];
    float ms = 0.0f;
    for (int c = 0; c < cols; ++c) ms += row[c] * row[c];
    ms /= cols;
    const float rstd = 1.0f / std::sqrt(ms + eps);
    for (int c = 0; c < cols; ++c)
      y[(size_t)r * cols + c] = row[c] * rstd * gamma[c];
  }
  return y;
}

void test_rmsnorm() {
  const int rows = 4, cols = 128;
  const auto x = ktest::randn(rows * cols, 0.0f, 1.0f);
  const auto gamma = ktest::randu(cols, 0.5f, 1.5f);

  ktest::DeviceArray<float> dx(rows * cols), dgamma(cols), dy(rows * cols);
  dx.upload(x);
  dgamma.upload(gamma);
  ck::rmsnorm_forward(dx.ptr, dgamma.ptr, dy.ptr, rows, cols);
  const auto got = dy.download();

  const auto ref = cpu_rmsnorm(x, gamma, rows, cols, 1e-5f);
  ktest::close("rmsnorm", got, ref, 1e-3f, 1e-4f);
}
