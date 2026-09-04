# CUDA Kernels & Operators

A self-contained library of hand-written CUDA kernels, each with a CPU-reference
unit test. The kernels are chosen to be a notch harder than textbook examples
(`reduceMax`, elementwise add, …) — several use block/warp reductions, shared
memory tiling, parallel primitives, and **operator fusion**.

## Layout

```
kernels/
├── CMakeLists.txt
├── include/kernels/      # public headers (host API)
├── src/                  # .cu implementations + common.cuh (device helpers)
│   └── common.cuh        # warp/block reductions, GELU
└── tests/                # one test_*.cu per kernel + self-contained harness
```

## The 12 kernels

| # | Kernel | What it demonstrates |
|---|--------|----------------------|
| 1 | `softmax` | numerically stable softmax via two-pass block reduction |
| 2 | `layernorm` | two-pass reduction (mean → variance) + affine |
| 3 | `rmsnorm` | single-pass reduction, modern LLM norm |
| 4 | `gelu` | elementwise tanh-approx activation |
| 5 | `fused_layernorm_gelu` | **fusion**: layernorm + GELU in one kernel |
| 6 | `fused_bias_gelu` | **fusion**: bias-add + GELU in one kernel |
| 7 | `matmul` | tiled GEMM (shared-memory 32×32 tiles) |
| 8 | `fused_linear` | **fusion**: GEMM + bias + GELU epilogue |
| 9 | `attention` | **fusion**: QKᵀ + online softmax + V in one sweep |
| 10 | `scan` | work-efficient Blelloch inclusive prefix sum |
| 11 | `topk` | per-thread sorted lists + shared-memory merge tournament |
| 12 | `rope` | rotary position embeddings |

Notes on the more involved ones:

- **`attention`** never materializes the `[T, T]` score matrix. Each block owns
  one query and sweeps keys once, maintaining a running max/sum to produce a
  numerically stable softmax on the fly (the "online softmax" idea behind
  FlashAttention). Requires `D == blockDim.x` (head dim up to 1024).
- **`scan`** is a single-block Blelloch up-sweep/down-sweep; `n` must be a power
  of two equal to `blockDim.x` (e.g. 512 / 1024).
- **`topk`** requires `k <= 32`.

## Build & run

From the repo root (CUDA toolkit + a driver-capable GPU required):

```bash
cmake -S . -B build
cmake --build build -j
./build/kernels/kernel_tests
```

or standalone:

```bash
cmake -S kernels -B build
cmake --build build -j
./build/kernel_tests
```

Override the target architecture if your GPU is newer than the defaults
(sm_70–sm_89):

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90
```

## Tests

Each `tests/test_*.cu` generates deterministic host inputs, runs the kernel,
downloads the result, and compares against a naive CPU reference with an
element-wise `|got − ref| ≤ atol + rtol·|ref|` check. The runner reports a
pass/fail summary and the max absolute error per test. There are no external
test-framework dependencies.
