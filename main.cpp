#include <iostream>
#include "matmul.h"
#include "timeutils.h"
#include "matrix.h"
#include <thread>

#include <fstream>

#define STEP_SIZE 24

int main()
{
    std::ofstream outputFile("benchmark.csv");

    if (!outputFile)
    {
        std::cerr << "Error opening the file." << std::endl;
        return 1;
    }

    // Write header
    // outputFile << "Model"<<",";
    for (int i = 6; i < 80; i += 5)
    {
        outputFile << STEP_SIZE * i << ",";
    }


    for (int i = 6; i < 120; i += 4)
    {
        Matrix A(STEP_SIZE * i, STEP_SIZE * i);
        Matrix B(STEP_SIZE * i, STEP_SIZE * i);
        Matrix out(STEP_SIZE * i, STEP_SIZE * i);

        auto gflops = calculate_gflops(A, B, out, naive_matmul_4x4_neon_3);
        std::cout << STEP_SIZE * i << " gflops: " << gflops << std::endl;
        outputFile<< gflops<<",";
    }
    outputFile << std::endl;
}
