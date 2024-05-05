#include <iostream>
#include "matmul.h"
#include "timeutils.h"
#include "matrix.h"
#include <thread>


int main()
{
    Matrix A(224, 224);
    Matrix B(224, 224);
    Matrix out(224, 224);

    auto At = A.transpose();

    auto gflops = calculate_gflops(At, B, out, neon_matmul);
    std::cout << "Gflops: " << gflops << std::endl;
}
