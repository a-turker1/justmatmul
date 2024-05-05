#include <iostream>
#include "matmul.h"
#include "timeutils.h"
#include "matrix.h"
#include <thread>


int main()
{
    Matrix A(48 * 12, 256);
    Matrix B(256, 48 * 12);
    Matrix out(48 * 12, 48 * 12);

    auto At = A.transpose();

    auto gflops = calculate_gflops(At, B, out, neon_matmul);
    std::cout << "Gflops: " << gflops << std::endl;
}
