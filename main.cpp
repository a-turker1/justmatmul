#include <iostream>
#include "matmul.h"
#include "timeutils.h"
#include "matrix.h"
#include <thread>


int main()
{
    Matrix A(8, 12);
    Matrix B(8, 8);
    Matrix out(12, 8);

    auto gflops = calculate_gflops(A, B, out, neon_matmul);
    std::cout << "Gflops: " << gflops << std::endl;
    std::cout << out << std::endl;
}
