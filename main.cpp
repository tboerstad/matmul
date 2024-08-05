#define PICOBENCH_IMPLEMENT
#include "picobench/picobench.hpp"

#include "matrix.h"
#include <numeric>
#include <omp.h>
#include <vector>

const int N = 500;

static void mat_mul_naive_v2(const Matrix &a, const Matrix &b, Matrix &out) {
  const int rowsOut = a.rows();
  const int colsOut = b.cols();
  const int innerDim = a.cols();
  for (int i = 0; i < rowsOut; ++i) {
    for (int j = 0; j < colsOut; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < innerDim; ++k) {
        sum += a(i, k) * b(k, j);
      }
      out(i, j) = sum;
    }
  }
}

static void mat_mul_naive(const Matrix &a, const Matrix &b, Matrix &out) {
  const int rowsOut = a.rows();
  const int colsOut = b.cols();
  const int innerDim = a.cols();
  for (int i = 0; i < rowsOut; ++i) {
    for (int j = 0; j < colsOut; ++j) {
      for (int k = 0; k < innerDim; ++k) {
        out(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

static void mat_mul_cache(const Matrix &a, const Matrix &b, Matrix &out) {
  const int rowsOut = a.rows();
  const int colsOut = b.cols();
  const int innerDim = a.cols();
  for (int i = 0; i < rowsOut; ++i) {
    for (int k = 0; k < innerDim; ++k) {
      for (int j = 0; j < colsOut; ++j) {
        out(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

static void mat_mul_cache_omp(const Matrix &a, const Matrix &b, Matrix &out) {
  const int rowsOut = a.rows();
  const int colsOut = b.cols();
  const int innerDim = a.cols();

#pragma omp parallel for collapse(2)
  for (int i = 0; i < rowsOut; ++i) {
    for (int k = 0; k < innerDim; ++k) {
      for (int j = 0; j < colsOut; ++j) {
        out(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

static void mat_mul_wrapper(picobench::state &s,
                            void (*matmul_func)(const Matrix &, const Matrix &,
                                                Matrix &)) {
  srand(42);
  Matrix A(N, N);
  for (int i = 0; i < A.rows() * A.cols(); ++i) {
    A.data()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  Matrix B(N, N);
  for (int i = 0; i < B.rows() * A.cols(); ++i) {
    B.data()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  Matrix C(N, N);
  std::fill(C.data(), C.data() + N * N, 0.0f);

  bool first_iteration = true;

  for (auto _ : s) {
    matmul_func(A, B, C);

    if (first_iteration) {
      // FYI leaks memory
      float *flat_matrix = new float[N * N];
      std::copy(C.data(), C.data() + N * N, flat_matrix);
      s.set_result(reinterpret_cast<picobench::result_t>(flat_matrix));
      first_iteration = false;
    }
  }
}

static void mat_mul_naive_b(picobench::state &s) {
  mat_mul_wrapper(s, mat_mul_naive);
}
static void mat_mul_cache_b(picobench::state &s) {
  mat_mul_wrapper(s, mat_mul_cache);
}
static void mat_mul_cache_omp_b(picobench::state &s) {
  mat_mul_wrapper(s, mat_mul_cache_omp);
}
static void mat_mul_naive_v2_b(picobench::state &s) {
  mat_mul_wrapper(s, mat_mul_naive_v2);
}

PICOBENCH(mat_mul_naive_b);
PICOBENCH(mat_mul_naive_v2_b);
PICOBENCH(mat_mul_cache_b);
PICOBENCH(mat_mul_cache_omp_b);

bool compare_matrices(picobench::result_t a, picobench::result_t b) {

  float *mat_a = reinterpret_cast<float *>(a);
  float *mat_b = reinterpret_cast<float *>(b);
  float sum_squared =
      std::inner_product(mat_a, mat_a + N * N, mat_b, 0.0f, std::plus<>(),
                         [](float a, float b) { return (a - b) * (a - b); });

  //   std::copy(mat_a, mat_a + N * N, std::ostream_iterator<float>(std::cout, "
  //   ")); std::copy(mat_b, mat_b + N * N,
  //   std::ostream_iterator<float>(std::cout, " "));

  return std::sqrt(sum_squared / (N * N)) < 1e-6;
}

int main(int argc, char *argv[]) {
  picobench::runner runner;
  runner.parse_cmd_line(argc, argv);

  runner.set_compare_results_across_benchmarks(true);
  runner.run_benchmarks(123);
  auto report = runner.generate_report(compare_matrices);
  report.to_text(std::cout);

  return 0;
}