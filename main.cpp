#include <iostream>
#include "matmul.h"
#include "timeutils.h"
#include "matrix.h"
#include <thread>

#include <fstream>

#define STEP_SIZE 8

struct Function {
    std::string name;
    void (*func)(Matrix &, Matrix &, Matrix &);
};

int main()
{

    for (int i = 5; i < 300; i += 12)
    {
        Matrix A(STEP_SIZE * i, STEP_SIZE * i);
        Matrix B(STEP_SIZE * i, STEP_SIZE * i);
        Matrix out(STEP_SIZE * i, STEP_SIZE * i);

        auto gflops = calculate_gflops(A, B, out, matmul);
        std::cout << "Matrix size: " << STEP_SIZE *i <<"x" << STEP_SIZE *i << " Gflops: " << gflops << std::endl;
    } 
}
