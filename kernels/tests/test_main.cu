// Test runner: registers and runs every kernel unit test, then reports a
// pass/fail summary. Each test lives in its own test_*.cu file.

#include <cuda_runtime.h>

#include <cstdio>

#include "test_utils.hpp"

// Forward declarations (one per kernel).
void test_softmax();
void test_layernorm();
void test_rmsnorm();
void test_gelu();
void test_fused_layernorm_gelu();
void test_fused_bias_gelu();
void test_matmul();
void test_fused_linear();
void test_attention();
void test_scan();
void test_topk();
void test_rope();

int main() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("Running kernel unit tests on: %s (sm_%d%d)\n\n", prop.name,
              prop.major, prop.minor);

  test_softmax();
  test_layernorm();
  test_rmsnorm();
  test_gelu();
  test_fused_layernorm_gelu();
  test_fused_bias_gelu();
  test_matmul();
  test_fused_linear();
  test_attention();
  test_scan();
  test_topk();
  test_rope();

  CUDA_CHECK(cudaDeviceSynchronize());

  std::printf("\n%d checks, %d failure(s)\n", ktest::g_checks,
              ktest::g_failures);
  return ktest::g_failures == 0 ? 0 : 1;
}
