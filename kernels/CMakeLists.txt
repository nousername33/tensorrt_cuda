# CUDA kernel library + unit tests.
#
# Build from the repo root:
#   cmake -S . -B build && cmake --build build -j && ./build/kernels/kernel_tests
#
# Or standalone:
#   cmake -S kernels -B build && cmake --build build -j && ./build/kernel_tests

if(NOT CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to a broad set of common architectures; override with
#   -DCMAKE_CUDA_ARCHITECTURES=90
# (or whatever sm_xx your GPU uses) if your device is not covered below.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89" CACHE STRING
      "CUDA target architectures" FORCE)
endif()

# ---------------------------------------------------------------------------
# Kernel library
# ---------------------------------------------------------------------------
set(CUDA_KERNELS_SOURCES
  src/softmax.cu
  src/layernorm.cu
  src/rmsnorm.cu
  src/gelu.cu
  src/fused_layernorm_gelu.cu
  src/fused_bias_gelu.cu
  src/matmul.cu
  src/fused_linear.cu
  src/attention.cu
  src/scan.cu
  src/topk.cu
  src/rope.cu
)

add_library(cuda_kernels STATIC ${CUDA_KERNELS_SOURCES})
target_include_directories(cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
set(CUDA_KERNELS_TESTS
  tests/test_main.cu
  tests/test_softmax.cu
  tests/test_layernorm.cu
  tests/test_rmsnorm.cu
  tests/test_gelu.cu
  tests/test_fused_layernorm_gelu.cu
  tests/test_fused_bias_gelu.cu
  tests/test_matmul.cu
  tests/test_fused_linear.cu
  tests/test_attention.cu
  tests/test_scan.cu
  tests/test_topk.cu
  tests/test_rope.cu
)

add_executable(kernel_tests ${CUDA_KERNELS_TESTS})
target_include_directories(kernel_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
target_link_libraries(kernel_tests PRIVATE cuda_kernels)

enable_testing()
add_test(NAME kernel_tests COMMAND kernel_tests)
