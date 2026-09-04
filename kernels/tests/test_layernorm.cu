#include "kernels/layernorm.h"
#include "test_utils.hpp"

#include <cmath>

static std::vector<float> cpu_layernorm(const std::vector<float>& x,
                                        const std::vector<float>& gamma,
                                        const std::vector<float>& beta, int rows,
                                        int cols, float eps) {
  std::vector<float> y(x.size());
  for (int r = 0; r < rows; ++r) {
    const float* row = &x[(size_t)r * cols];
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) mean += row[c];
    mean /= cols;
    float var = 0.0f;
    for (int c = 0; c < cols; ++c) {
      const float d = row[c] - mean;
      var += d * d;
    }
    var /= cols;
    const float rstd = 1.0f / std::sqrt(var + eps);
    for (int c = 0; c < cols; ++c)
      y[(size_t)r * cols + c] = (row[c] - mean) * rstd * gamma[c] + beta[c];
  }
  return y;
}

void test_layernorm() {
  const int rows = 4, cols = 128;
  const auto x = ktest::randn(rows * cols, 0.0f, 1.0f);
  const auto gamma = ktest::randu(cols, 0.5f, 1.5f);
  const auto beta = ktest::randu(cols, -0.5f, 0.5f);

  ktest::DeviceArray<float> dx(rows * cols), dgamma(cols), dbeta(cols),
      dy(rows * cols);
  dx.upload(x);
  dgamma.upload(gamma);
  dbeta.upload(beta);
  ck::layernorm_forward(dx.ptr, dgamma.ptr, dbeta.ptr, dy.ptr, rows, cols);
  const auto got = dy.download();

  const auto ref = cpu_layernorm(x, gamma, beta, rows, cols, 1e-5f);
  ktest::close("layernorm", got, ref, 1e-3f, 1e-4f);
}
