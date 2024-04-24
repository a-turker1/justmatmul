#include <iostream>
#include "matmul.h"
#include "timeutils.h"
#include "matrix.h"

int main(){
    Matrix A(128,64);
    Matrix B(64,128);
    Matrix out(128,128);
    naive_matmul(A, B, out);
    auto flops = calculate_gflops(A,B,out, naive_matmul);
    std::cout << flops << std::endl;

}
