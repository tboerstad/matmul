# Matrix Multiplication CPU Performance Investigation

This repository contains a performance investigation of various matrix multiplication implementations on CPU. The project uses [picobench](https://github.com/iboB/picobench) for benchmarking.

## Overview

This investigation compares the performance of different matrix multiplication algorithms:

1. Naive implementation
2. Naive implementation (version 2)
3. Cache-friendly implementation
4. Cache-friendly implementation with OpenMP parallelization

## Results

The benchmarks were run on an Apple M1 Pro processor. Here are the results:

| Name (* = baseline)      | Dim | Total ms | ns/op    | Baseline | Ops/second |
|--------------------------|-----|----------|----------|----------|------------|
| mat_mul_naive_b *        | 10  | 1891.359 | 189135e3 | -        | 5.3        |
| mat_mul_naive_v2_b       | 10  | 1729.368 | 172936e3 | 0.914    | 5.8        |
| mat_mul_cache_b          | 10  | 126.243  | 12624333 | 0.067    | 79.2       |
| mat_mul_cache_omp_b      | 10  | 28.014   | 2801441  | 0.015    | 357.0      |

Note: 
- The 'Dim' column indicates that each implementation is run 10 times.
- Each run performs multiplication of 500x500 matrices.

## Implementations

The repository includes the following matrix multiplication implementations:

1. `mat_mul_naive`: A basic triple-nested loop implementation
2. `mat_mul_naive_v2`: A slightly optimized version of the naive implementation
3. `mat_mul_cache`: A cache-friendly implementation with reordered loops
4. `mat_mul_cache_omp`: A cache-friendly implementation using OpenMP for parallelization

## Building and Running

To run the benchmarks:

```bash
make run
```

## Dependencies

- [picobench](https://github.com/iboB/picobench): A micro-benchmarking library for C++
- OpenMP: Used for parallelization in the `mat_mul_cache_omp` implementation
- Clang compiler with OpenMP support
- Homebrew-installed OpenMP library (for macOS)

Note: The Makefile assumes you have OpenMP installed via Homebrew on macOS. If you're using a different system or setup, you may need to adjust the compiler flags in the Makefile.