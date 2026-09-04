#include "kernels/fused_linear.h"
#include "test_utils.hpp"

#include <cmath>

static float cpu_gelu(float x) {
  const float c = 0.7978845608028654f;
  return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
}

static std::vector<float> cpu_ref(const std::vector<float>& A,
                                  const std::vector<float>& B,
                                  const std::vector<float>& bias, int M, int N,
                                  int K) {
  std::vector<float> C((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
      C[(size_t)m * N + n] = cpu_gelu(acc + bias[n]);
    }
  return C;
}

void test_fused_linear() {
  const int M = 64, N = 48, K = 32;
  const auto A = ktest::randu(M * K, -1.0f, 1.0f);
  const auto B = ktest::randu(K * N, -1.0f, 1.0f);
  const auto bias = ktest::randu(N, -0.5f, 0.5f);

  ktest::DeviceArray<float> dA(M * K), dB(K * N), dbias(N), dC(M * N);
  dA.upload(A);
  dB.upload(B);
  dbias.upload(bias);
  ck::fused_linear_gelu_forward(dA.ptr, dB.ptr, dbias.ptr, dC.ptr, M, N, K);
  const auto got = dC.download();

  const auto ref = cpu_ref(A, B, bias, M, N, K);
  ktest::close("fused_linear_gelu", got, ref, 1e-3f, 1e-3f);
}
