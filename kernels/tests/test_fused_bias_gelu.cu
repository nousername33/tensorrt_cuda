#include "kernels/fused_bias_gelu.h"
#include "test_utils.hpp"

#include <cmath>

static float cpu_gelu(float x) {
  const float c = 0.7978845608028654f;
  return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
}

void test_fused_bias_gelu() {
  const int rows = 32, cols = 64;
  const auto x = ktest::randn(rows * cols, 0.0f, 1.0f);
  const auto bias = ktest::randu(cols, -1.0f, 1.0f);

  ktest::DeviceArray<float> dx(rows * cols), dbias(cols), dy(rows * cols);
  dx.upload(x);
  dbias.upload(bias);
  ck::fused_bias_gelu_forward(dx.ptr, dbias.ptr, dy.ptr, rows, cols);
  const auto got = dy.download();

  std::vector<float> ref(x.size());
  for (int i = 0; i < rows * cols; ++i)
    ref[i] = cpu_gelu(x[i] + bias[i % cols]);
  ktest::close("fused_bias_gelu", got, ref, 1e-4f, 1e-6f);
}
