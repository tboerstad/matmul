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

 Name (* = baseline)      |   Dim   |  Total ms |  ns/op  |Baseline| Ops/second
--------------------------|--------:|----------:|--------:|-------:|----------:
 mat_mul_naive_b *        |       1 |  1730.107 |173010e4 |      - |        0.6
 mat_mul_naive_acc_b      |       1 |  1645.850 |164585e4 |  0.951 |        0.6
 mat_mul_cache_b          |       1 |   107.201 |107201e3 |  0.062 |        9.3
 mat_mul_simd_b           |       1 |    45.826 |45825750 |  0.026 |       21.8
 mat_mul_simd_advanced_b  |       1 |    17.634 |17633750 |  0.010 |       56.7
 mat_mul_cache_omp_b      |       1 |    16.527 |16527500 |  0.010 |       60.5
 mat_mul_eigen_b          |       1 |     9.785 | 9785125 |  0.006 |      102.2

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