#include <iostream>
#include "matmul.h"
#include "allocator.h"
#include "matrix.h"

int main(){
    float *src = allocate(3,2);
    Matrix A(10,5);
    Matrix B(5,7);
    Matrix out(10,7);
    naive_matmul(A, B, out);
    std::cout << out << std::endl;
}
