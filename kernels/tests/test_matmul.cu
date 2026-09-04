#include "kernels/matmul.h"
#include "test_utils.hpp"

static std::vector<float> cpu_matmul(const std::vector<float>& A,
                                     const std::vector<float>& B, int M, int N,
                                     int K) {
  std::vector<float> C((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
      C[(size_t)m * N + n] = acc;
    }
  return C;
}

void test_matmul() {
  const int M = 64, N = 48, K = 32;
  const auto A = ktest::randu(M * K, -1.0f, 1.0f);
  const auto B = ktest::randu(K * N, -1.0f, 1.0f);

  ktest::DeviceArray<float> dA(M * K), dB(K * N), dC(M * N);
  dA.upload(A);
  dB.upload(B);
  ck::matmul_forward(dA.ptr, dB.ptr, dC.ptr, M, N, K);
  const auto got = dC.download();

  const auto ref = cpu_matmul(A, B, M, N, K);
  ktest::close("matmul", got, ref, 1e-3f, 1e-3f);
}
