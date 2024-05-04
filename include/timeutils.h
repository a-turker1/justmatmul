#include <chrono>
#include "matrix.h"

template <typename Op>
double calculate_gflops(Matrix& A, Matrix& B, Matrix& out, Op op){
    double elapsed_time = 0;
    for(size_t i = 0; i < 5; i++)
        op(A, B, out);

    for (size_t i = 0; i < 20; i++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        op(A, B, out);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        elapsed_time += duration.count();
    }
    elapsed_time /= 20.0;
    
    int M = A.cols;
    int N = A.rows;
    int K = B.rows;
    long long total_flops = 2LL * M * K * N;
    std::cout << elapsed_time<<std::endl;
    std::cout << total_flops<<std::endl;
    double gflops = total_flops / elapsed_time;
    return gflops;
}
