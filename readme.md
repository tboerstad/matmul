# Matrix Multiplication CPU Performance Investigation

This repository contains a performance investigation of various matrix multiplication implementations on CPU. The project uses [picobench](https://github.com/iboB/picobench) for benchmarking.

## Overview

This investigation compares the performance of different matrix multiplication algorithms:

1. Naive implementation
2. Naive implementation with accumulator
3. Cache-friendly implementation
4. SIMD implementation
5. Cache-friendly implementation with OpenMP parallelization
6. Advanced SIMD implementation
7. Eigen library implementation

## Results

The benchmarks were run on an Apple M1 Pro processor. Here are the latest results:

| Name (* = baseline)      | Dim | Total ms | ns/op    | Baseline | Ops/second |
|--------------------------|-----|----------|----------|----------|------------|
| mat_mul_naive_b *        | 1   | 1208.279 | 120827e4 | -        | 0.8        |
| mat_mul_naive_acc_b      | 1   | 1159.172 | 115917e4 | 0.959    | 0.9        |
| mat_mul_cache_b          | 1   | 78.609   | 78609500 | 0.065    | 12.7       |
| mat_mul_simd_b           | 1   | 39.607   | 39607417 | 0.033    | 25.2       |
| mat_mul_cache_omp_b      | 1   | 16.121   | 16121416 | 0.013    | 62.0       |
| mat_mul_simd_advanced_b  | 1   | 11.434   | 11433542 | 0.009    | 87.5       |
| mat_mul_eigen_b          | 1   | 11.023   | 11022584 | 0.009    | 90.7       |

Note: 
- All benchmarks were performed on an Apple M1 Pro processor.
- The 'Dim' column indicates that each implementation is run once.
- Each run performs multiplication of 1000x1000 matrices (N = 1000 in the code).

## Implementations

The repository includes the following matrix multiplication implementations:

1. `mat_mul_naive`: A basic triple-nested loop implementation
2. `mat_mul_naive_acc`: A slightly optimized version of the naive implementation using an accumulator
3. `mat_mul_cache`: A cache-friendly implementation with reordered loops
4. `mat_mul_simd`: An implementation using SIMD instructions (ARM NEON)
5. `mat_mul_cache_omp`: A cache-friendly implementation using OpenMP for parallelization
6. `mat_mul_simd_advanced`: An advanced implementation combining SIMD and cache optimization techniques
7. `mat_mul_eigen`: An implementation using the Eigen linear algebra library

## Building and Running

To run the benchmarks:

```bash
make run
```

## Dependencies

- [picobench](https://github.com/iboB/picobench): A micro-benchmarking library for C++
- OpenMP: Used for parallelization in various implementations
- Eigen: A C++ template library for linear algebra
- Clang compiler with OpenMP support
- Homebrew-installed OpenMP and Eigen libraries (for macOS)

Note: The Makefile assumes you have OpenMP and Eigen installed via Homebrew on macOS. If you're using a different system or setup, you may need to adjust the compiler flags in the Makefile.