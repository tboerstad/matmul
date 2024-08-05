#define PICOBENCH_IMPLEMENT
#include "picobench/picobench.hpp"

#include "matrix.h"
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Dense>
#include <arm_neon.h>
#include <numeric>
#include <omp.h>
#include <vector>

#define CREATE_BENCHMARK(func_name) \
static void func_name##_b(picobench::state &s) { \
  mat_mul_wrapper(s, func_name); \
} \
PICOBENCH(func_name##_b);

const int N = 500;

static void mat_mul_naive_acc(const Matrix &a, const Matrix &b, Matrix &out) {
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

static void mat_mul_simd(const Matrix &a, const Matrix &b, Matrix &out) {
    const int M = a.rows();
    const int N = b.cols();
    const int K = a.cols();
    constexpr int VECTOR_SIZE = 4; // NEON processes 4 floats at a time
    constexpr int BLOCK_SIZE = 64; // Adjust based on cache size

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int ii = i; ii < std::min(i + BLOCK_SIZE, M); ++ii) {
                for (int jj = j; jj < std::min(j + BLOCK_SIZE, N); jj += VECTOR_SIZE) {
                    float32x4_t sum = vdupq_n_f32(0.0f);
                    for (int k = 0; k < K; ++k) {
                        float32x4_t b_vec = vld1q_f32(&b(k, jj));
                        float32x4_t a_val = vdupq_n_f32(a(ii, k));
                        sum = vfmaq_f32(sum, a_val, b_vec);
                    }
                    vst1q_f32(&out(ii, jj), sum);
                }
            }
        }
    }
}

static void mat_mul_simd_advanced(const Matrix &a, const Matrix &b, Matrix &out) {
    const int M = a.rows();
    const int N = b.cols();
    const int K = a.cols();
    constexpr int VECTOR_SIZE = 4; // NEON processes 4 floats at a time
    constexpr int BLOCK_SIZE = 64; // Adjust based on cache size

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < K; k += BLOCK_SIZE) {
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, M); ii += VECTOR_SIZE) {
                    for (int jj = j; jj < std::min(j + BLOCK_SIZE, N); jj += VECTOR_SIZE) {
                        float32x4_t sum[VECTOR_SIZE] = {
                            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
                        };
                        for (int kk = k; kk < std::min(k + BLOCK_SIZE, K); ++kk) {
                            float32x4_t b_vec = vld1q_f32(&b(kk, jj));
                            for (int v = 0; v < VECTOR_SIZE; ++v) {
                                float32x4_t a_val = vdupq_n_f32(a(ii + v, kk));
                                sum[v] = vfmaq_f32(sum[v], a_val, b_vec);
                            }
                        }
                        for (int v = 0; v < VECTOR_SIZE; ++v) {
                            float32x4_t current = vld1q_f32(&out(ii + v, jj));
                            vst1q_f32(&out(ii + v, jj), vaddq_f32(current, sum[v]));
                        }
                    }
                }
            }
        }
    }
}

static void mat_mul_eigen(const Matrix &a, const Matrix &b, Matrix &out) {
  // Convert custom Matrix to Eigen::MatrixXf using Map
  Eigen::Map<const Eigen::MatrixXf> eigenA(a.data(), a.rows(), a.cols());
  Eigen::Map<const Eigen::MatrixXf> eigenB(b.data(), b.rows(), b.cols());

  // Map the output Matrix to Eigen::MatrixXf
  Eigen::Map<Eigen::MatrixXf> eigenOut(out.data(), out.rows(), out.cols());

  // Perform matrix multiplication directly into the output
  eigenOut.noalias() = eigenA * eigenB;
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

CREATE_BENCHMARK(mat_mul_naive);
CREATE_BENCHMARK(mat_mul_naive_acc);
CREATE_BENCHMARK(mat_mul_cache);
CREATE_BENCHMARK(mat_mul_simd);
CREATE_BENCHMARK(mat_mul_cache_omp);
CREATE_BENCHMARK(mat_mul_eigen);
CREATE_BENCHMARK(mat_mul_simd_advanced);


bool compare_matrices(picobench::result_t a, picobench::result_t b) {

  float *mat_a = reinterpret_cast<float *>(a);
  float *mat_b = reinterpret_cast<float *>(b);
  float sum_squared =
      std::inner_product(mat_a, mat_a + N * N, mat_b, 0.0f, std::plus<>(),
                         [](float a, float b) { return (a - b) * (a - b); });
    std::cout << std::setprecision(9) << std::fixed << mat_b[(N*N)/2] << std::endl;
  return std::sqrt(sum_squared / (N * N)) < 1e-3f;
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
