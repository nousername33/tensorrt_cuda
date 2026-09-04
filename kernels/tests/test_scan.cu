#include "kernels/scan.h"
#include "test_utils.hpp"

static std::vector<float> cpu_scan(const std::vector<float>& x, int n) {
  std::vector<float> y(n);
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) {
    acc += x[i];
    y[i] = acc;
  }
  return y;
}

void test_scan() {
  const int n = 1024;  // power of two, equals blockDim.x
  const auto x = ktest::randu(n, -1.0f, 1.0f);

  ktest::DeviceArray<float> dx(n), dy(n);
  dx.upload(x);
  ck::scan_inclusive_forward(dx.ptr, dy.ptr, n);
  const auto got = dy.download();

  const auto ref = cpu_scan(x, n);
  ktest::close("scan", got, ref, 1e-3f, 1e-2f);
}
