#include <iostream>
#include "matmul.h"
#include "timeutils.h"
#include "matrix.h"
#include <thread>

#include <fstream>

#define STEP_SIZE 24

struct Function {
    std::string name;
    void (*func)(Matrix &, Matrix &, Matrix &);
};

int main()
{
    std::ofstream outputFile("benchmark.csv");

    if (!outputFile)
    {
        std::cerr << "Error opening the file." << std::endl;
        return 1;
    }

    std::vector<Function> benchmarks;
    benchmarks.push_back({"naive_matmul_4x4_neon", naive_matmul_4x4_neon});
    benchmarks.push_back({"naive_matmul_4x4_neon_2", naive_matmul_4x4_neon_2});
    benchmarks.push_back({"naive_matmul_4x4_neon_3", naive_matmul_4x4_neon_3});
    benchmarks.push_back({"naive_matmul_4x4_neon_4", naive_matmul_4x4_neon_4});
    benchmarks.push_back({"naive_matmul_4x4_neon_5", naive_matmul_4x4_neon_5});



    // Write header
    outputFile << "Model"<<",";
    for (int i = 6; i < 92; i += 4)
    {
        outputFile << STEP_SIZE * i << ",";
    }
    outputFile << std::endl;
    
    for (const auto &benchmark : benchmarks) {
        outputFile<<benchmark.name<<",";
        for (int i = 6; i < 92; i += 4)
        {
            Matrix A(STEP_SIZE * i, STEP_SIZE * i);
            Matrix B(STEP_SIZE * i, STEP_SIZE * i);
            Matrix out(STEP_SIZE * i, STEP_SIZE * i);

            auto gflops = calculate_gflops(A, B, out, benchmark.func);
            std::cout << STEP_SIZE * i << " gflops: " << gflops << std::endl;
            outputFile<< gflops<<",";
        }
        outputFile << std::endl;
    }

    outputFile << std::endl;
}
