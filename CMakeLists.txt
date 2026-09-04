cmake_minimum_required(VERSION 3.18)

# Root build for the tensorrt_cuda study repo. The kernels/ subdirectory
# contains a self-contained CUDA kernel library + unit tests; it can also be
# configured and built on its own (cmake -S kernels -B build).
project(tensorrt_cuda LANGUAGES CXX CUDA)

add_subdirectory(kernels)
