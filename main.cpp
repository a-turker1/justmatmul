#include <iostream>
#include "matmul.h"
#include "timeutils.h"
#include "matrix.h"
#include <thread>


int main()
{
    Matrix A(36, 64);
    Matrix B(64, 64);
    Matrix out(36, 64);

    auto At = A.transpose();

    auto gflops = calculate_gflops(At, B, out, neon_matmul);
    std::cout << "Gflops: " << gflops << std::endl;
}
