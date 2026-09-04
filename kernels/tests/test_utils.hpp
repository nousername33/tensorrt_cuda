#pragma once

// Minimal, dependency-free test harness for the CUDA kernel unit tests.
// No GoogleTest / Catch2 required: each test is a `void test_xxx()` function
// that uses KT_CHECK / close() and is registered in test_main.cu.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace ktest {

inline int g_failures = 0;
inline int g_checks = 0;

inline void fail(const char* file, int line, const std::string& msg) {
  ++g_failures;
  std::fprintf(stderr, "  [FAIL] %s:%d  %s\n", file, line, msg.c_str());
}

#define KT_CHECK(cond)                                        \
  do {                                                       \
    ++ktest::g_checks;                                       \
    if (!(cond)) ktest::fail(__FILE__, __LINE__, #cond);     \
  } while (0)

// Check a CUDA API call succeeded; abort the test process on error so the
// stack of failures is not flooded by a cascade.
#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
      std::fprintf(stderr, "  [CUDA] %s:%d  %s: %s\n", __FILE__,     \
                   __LINE__, #call, cudaGetErrorString(_e));          \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

// ---------------------------------------------------------------------------
// Random host input generation (deterministic via a fixed seed).
// ---------------------------------------------------------------------------
inline std::vector<float> randu(size_t n, float lo, float hi, unsigned seed = 1234) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& e : v) e = d(gen);
  return v;
}

inline std::vector<float> randn(size_t n, float mean, float stddev,
                                unsigned seed = 1234) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> d(mean, stddev);
  std::vector<float> v(n);
  for (auto& e : v) e = d(gen);
  return v;
}

// ---------------------------------------------------------------------------
// RAII device buffer with upload/download helpers.
// ---------------------------------------------------------------------------
template <typename T>
struct DeviceArray {
  T* ptr = nullptr;
  size_t n = 0;

  explicit DeviceArray(size_t count) : n(count) {
    if (count) CUDA_CHECK(cudaMalloc((void**)&ptr, count * sizeof(T)));
  }
  ~DeviceArray() {
    if (ptr) cudaFree(ptr);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  void upload(const std::vector<T>& h) {
    CUDA_CHECK(cudaMemcpy(ptr, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

// ---------------------------------------------------------------------------
// Floating point comparison: every element must satisfy
//   |got - ref| <= atol + rtol * |ref|
// Prints the max absolute error on success.
// ---------------------------------------------------------------------------
inline void close(const char* name, const std::vector<float>& got,
                  const std::vector<float>& ref, float rtol, float atol) {
  ++g_checks;
  if (got.size() != ref.size()) {
    fail(__FILE__, __LINE__,
         std::string(name) + ": size mismatch " + std::to_string(got.size()) +
             " vs " + std::to_string(ref.size()));
    return;
  }
  float maxerr = 0.0f;
  size_t maxi = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float e = std::fabs(got[i] - ref[i]);
    if (e > maxerr) {
      maxerr = e;
      maxi = i;
    }
    const float tol = atol + rtol * std::fabs(ref[i]);
    if (e > tol) {
      fail(__FILE__, __LINE__,
           std::string(name) + " @ [" + std::to_string(i) + "]: got " +
               std::to_string(got[i]) + " ref " + std::to_string(ref[i]) +
               " (err " + std::to_string(e) + " > tol " + std::to_string(tol) +
               ")");
      return;
    }
  }
  std::printf("  [OK] %-24s (max abs err %.3e @ %zu)\n", name, maxerr, maxi);
}

}  // namespace ktest
