#include <chrono>
#include "matrix.h"

template <typename Op>
double calculate_gflops(Matrix& A, Matrix& B, Matrix& out, Op op){
    auto start_time = std::chrono::high_resolution_clock::now();
    op(A, B, out);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double elapsed_time = duration.count() / 1000.0; // Convert to seconds
    int M = A.cols;
    int N = B.rows;
    int K = out.rows;
    long long total_flops = 2LL * M * K * N;
    std::cout << elapsed_time<<std::endl;
    double gflops = total_flops / (elapsed_time * 1e9);
    return gflops;
}