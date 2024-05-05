#include "matmul.h"
#include <arm_neon.h>
#include "micro.h"

void naive_matmul(Matrix &a, Matrix &b, Matrix &out)
{
    int M = a.rows;
    int K = a.cols;
    int N = b.rows;
    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            float b = bData[k + n * K];
            for (int m = 0; m < M; m++)
            {
                outData[m + m * N] += aData[m + k * M] * b;
            }
        }
    }
}

void neon_matmul(Matrix &a, Matrix &b, Matrix &out)
{
    int N = a.rows;
    int M = a.cols;
    int K = b.cols;
    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

    for (int m = 0; m < M - 11; m +=12)
    {
        for (int k = 0; k < K - 7; k +=8)
        {
            matmul_12x8_micro_kernel(N, aData + m, bData + k, outData + m * out.cols + k, a.cols, b.cols, out.cols);
        }
    }

    int remaining_M = M % 12;
    int remaining_K = K % 8;


    for (size_t m = 0; m < M-3; m += 4)
    {
        for (size_t k = K - remaining_K; k < K-3; k += 4)
        {
            matmul_4x4_micro_kernel(N, aData + m, bData + k, outData + m * out.cols + k, a.cols, b.cols, out.cols);
        }
    }

    for (size_t m = M - remaining_M; m < M-3; m += 4)
    {
        for (size_t k = 0; k < K - remaining_K -3; k += 4)
        {
            matmul_4x4_micro_kernel(N, aData + m, bData + k, outData + m * out.cols + k, a.cols, b.cols, out.cols);
        }
    }

    
}