#include "matmul.h"

void naive_matmul(Matrix &a, Matrix &b, Matrix &out)
{
    int M = a.cols;
    int N = a.rows;
    int K = b.rows;
    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int n = 0; n < N; n++)
            {
                outData[m * K + k] += aData[m * N + n] * bData[n * K + k];
            }
        }
    }
}