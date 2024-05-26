#include "matmul.h"
#include <arm_neon.h>
#include "micro.h"

int min(float a, float b)
{
    return (a < b) ? a : b;
}

void naive_matmul(Matrix &a, Matrix &b, Matrix &out)
{
    int M = a.rows;
    int N = b.rows;
    int K = b.cols;

    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();
    int block_size = 128;

    for (int n = 0; n < N; n += block_size)
    {
        int n_ = min(N - n, block_size);
        naive_matmul_kernel(M, n_, K, aData , bData, outData, M, K, M);
    }
}

void matmul_4x4_neon(Matrix &a, Matrix &b_, Matrix &out)
{
    int M = a.rows;
    int N = a.cols;
    int K = b_.cols;
    
    auto b = b_.transpose();

    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();
    int block_size = 128;

    auto blocked_func = [&](int M, int N, int K, float *aData, int lda, float *bData, int ldb, float *outData, int ldo)
    {
        for (int m = 0; m < M - 3; m += 4)
        {
            for (int k = 0; k < K - 3; k += 4)
            {
                matmul_4x4_micro_kernel_row_major(N, aData + m, bData + k, outData + m + ldo * k, lda, ldb, ldo);
            }
        }
    };

    for (int n = 0; n < N; n += block_size)
    {
        int n_ = min(N - n, block_size);
        blocked_func(M, n_, K, aData + (M * n), M, bData + (K * n), K, outData, M);
    }
}



void matmul_12x8_neon(Matrix &a, Matrix &b_, Matrix &out)
{
    int M = a.rows;
    int N = a.cols;
    int K = b_.cols;
    
    auto b = b_.transpose();

    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();
    int block_size = 128;

    auto blocked_func = [&](int M, int N, int K, float *aData, int lda, float *bData, int ldb, float *outData, int ldo)
    {
        for (int m = 0; m < M - 11; m += 12)
        {
            for (int k = 0; k < K - 7; k += 8)
            {
                matmul_12x8_micro_kernel_col_major(N, aData + m, bData +  k, outData + m + ldo * k, lda, ldb, ldo);
            }
        }
    };

    for (int n = 0; n < N; n += block_size)
    {
        int n_ = min(N - n, block_size);
        blocked_func(M, n_, K, aData + (M * n), M, bData + (K * n), K, outData, M);
    }
}

void matmul(Matrix &a, Matrix &b_, Matrix &out){
    int M = a.rows;
    int N = a.cols;
    int K = b_.cols;
    
    auto b = b_.transpose();
    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

    auto matmul_12x8 = [&](int M, int N, int K, float *aData, int lda, float *bData, int ldb, float *outData, int ldo)
    {
        for (int m = 0; m < M - 11; m += 12)
        {
            for (int k = 0; k < K - 7; k += 8)
            {
                matmul_12x8_micro_kernel_col_major(N, aData + m, bData +  k, outData + m + ldo * k, lda, ldb, ldo);
            }
        }
    };

    auto matmul_4x4 = [&](int M, int N, int K, float *aData, int lda, float *bData, int ldb, float *outData, int ldo)
    {
        for (int m = 0; m < M - 3; m += 4)
        {
            for (int k = 0; k < K - 3; k += 4)
            {
                matmul_4x4_micro_kernel_row_major(N, aData + m, bData +  k, outData + m + ldo * k, lda, ldb, ldo);
            }
        }
    };


    auto naive_matmul = [&](int M, int N, int K, float *aData, int lda, float *bData, int ldb, float *outData, int ldo)
    {
        naive_matmul_kernel(M, N, K, aData, bData, outData, lda, ldb, ldo);
    };

    for (int n = 0; n < N; n += 128)
    {
        int n_ = min(N - n, 128);
        matmul_12x8(M, n_, K, aData + (M * n), M, bData + (K * n), K, outData, M);
    }

    int mleft = M % 12;
    int kleft = K % 8;

    for (int n = 0; n < N; n += 128)
    {
        int n_ = min(N - n, 128);
        matmul_4x4(M, n_, kleft, aData + (M * n), M, bData + (K * n) + (K - kleft), K, outData + (M * (K - kleft)), M);
    }

    for (int n = 0; n < N; n += 128)
    {
        int n_ = min(N - n, 128);
        matmul_4x4(mleft, n_, K - kleft, aData + (M * n) + (M - mleft), M, bData + (K * n) , K, outData + M - mleft, M);
    }

    mleft = M % 4;
    kleft = K % 4;

    for (int n = 0; n < N; n += 128)
    {
        int n_ = min(N - n, 128);
        naive_matmul(M, n_, kleft, aData + (M * n), M, bData + (K * n) + (K - kleft), K, outData + (M * (K - kleft)), M);
    }

    for (int n = 0; n < N; n += 128)
    {
        int n_ = min(N - n, 128);
        naive_matmul(mleft, n_, K - kleft, aData + (M * n) + (M - mleft), M, bData + (K * n) , K, outData + M - mleft, M);
    }


}