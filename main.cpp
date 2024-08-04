#define PICOBENCH_IMPLEMENT_WITH_MAIN
#include "picobench/picobench.hpp"

#include <vector>
#include "matrix.h"
#include <cstdlib> // for rand

static void mat_mul_naive(const Matrix& a, const Matrix& b, Matrix& out) {
    const int rowsOut = a.rows();
    const int colsOut = b.cols();
    const int innerDim = a.cols();
    for(int i = 0; i<rowsOut; ++i){
        for(int j=0; j<colsOut; ++j){
            for(int k=0; k<innerDim; ++k){
                out(i,j) += a(i,k)*b(k,j);
            }
        }
    }
}

static void mat_mul_cache(const Matrix& a, const Matrix& b, Matrix& out) {
    const int rowsOut = a.rows();
    const int colsOut = b.cols();
    const int innerDim = a.cols();
    for(int i = 0; i<rowsOut; ++i){
        for(int k=0; k<innerDim; ++k){
            for(int j=0; j<colsOut; ++j){
                out(i,j) += a(i,k)*b(k,j);
            }
        }
    }
}


static void mat_mul_wrapper(picobench::state& s, void (*matmul_func)(const Matrix&, const Matrix&, Matrix&)) {
    const int N = 300;
    Matrix A(N,N);
    for(int i=0; i<A.rows()*A.cols(); ++i){
        A.data()[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    Matrix B(N,N);
    for(int i=0; i<B.rows()*A.cols(); ++i){
        B.data()[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    Matrix C(N,N);
    for(int i=0; i<C.rows()*A.cols(); ++i){
        C.data()[i]=0;
    }
    for (auto _ : s)
    {
        matmul_func(A,B,C);
    }
}

static void mat_mul_naive_b(picobench::state& s) { mat_mul_wrapper(s, mat_mul_naive); }
static void mat_mul_cache_b(picobench::state& s) { mat_mul_wrapper(s, mat_mul_cache);}

PICOBENCH(mat_mul_naive_b);
PICOBENCH(mat_mul_cache_b);
