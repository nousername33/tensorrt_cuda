#include "kernels/gelu.h"
#include "test_utils.hpp"

#include <cmath>

static float cpu_gelu(float x) {
  const float c = 0.7978845608028654f;
  return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
}

void test_gelu() {
  const int n = 1 << 16;
  const auto x = ktest::randu(n, -5.0f, 5.0f);

  ktest::DeviceArray<float> dx(n), dy(n);
  dx.upload(x);
  ck::gelu_forward(dx.ptr, dy.ptr, n);
  const auto got = dy.download();

  std::vector<float> ref(n);
  for (int i = 0; i < n; ++i) ref[i] = cpu_gelu(x[i]);
  ktest::close("gelu", got, ref, 1e-4f, 1e-6f);
}
