#include <iostream>
#include "matmul.h"
#include "timeutils.h"
#include "matrix.h"

int main()
{
    Matrix A(1024, 1024);
    Matrix B(1024, 1024);
    Matrix out(1024, 1024);
    auto gflops = calculate_gflops(A, B, out, neon_matmul);
    std::cout << "Gflops: " << gflops << std::endl;
}
