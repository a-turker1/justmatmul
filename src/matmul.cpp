#include "matmul.h"
#include <arm_neon.h>
#include "micro.h"
void inner_loop(int N, int M, int K, float *aData, float *bData, float *outData, int lda, int ldb, int ldc);

void naive_matmul(Matrix &a, Matrix &b, Matrix &out)
{
    int M = a.rows;
    int N = b.rows;
    int K = b.cols;

    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k++)
        {
            float res = 0;
            for (int n = 0; n < N; n++)
            {
                res += aData[M * n + m] * bData[n + N * k];
            }
            outData[M * k + m] = res;
        }
    }
}

void naive_matmul_4x1(Matrix &a, Matrix &b, Matrix &out)
{
    int M = a.rows;
    int N = b.rows;
    int K = b.cols;

    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k++)
        {
            float res1 = 0;
            float res2 = 0;
            float res3 = 0;
            float res4 = 0;
            for (int n = 0; n < N; n += 4)
            {
                res1 += aData[M * n + m] * bData[n + N * k];
                res2 += aData[M * (n + 1) + m] * bData[1 + n + N * k];
                res3 += aData[M * (n + 2) + m] * bData[2 + n + N * k];
                res4 += aData[M * (n + 3) + m] * bData[3 + n + N * k];
            }
            outData[M * k + m] = res1 + res2 + res3 + res4;
        }
    }
}

void naive_matmul_4x4(Matrix &a, Matrix &b, Matrix &out)
{
    int M = a.rows;
    int N = b.rows;
    int K = b.cols;

    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k += 4)
        {
            float res1 = 0;
            float res2 = 0;
            float res3 = 0;
            float res4 = 0;
            float res5 = 0;
            float res6 = 0;
            float res7 = 0;
            float res8 = 0;
            float res9 = 0;
            float res10 = 0;
            float res11 = 0;
            float res12 = 0;
            float res13 = 0;
            float res14 = 0;
            float res15 = 0;
            float res16 = 0;
            for (int n = 0; n < N; n += 4)
            {
                res1 += aData[M * n + m] * bData[n + N * k];
                res2 += aData[M * (n + 1) + m] * bData[n + 1 + N * (k + 1)];
                res3 += aData[M * (n + 2) + m] * bData[n + 2 + N * (k + 2)];
                res4 += aData[M * (n + 3) + m] * bData[n + 3 + N * (k + 3)];

                res5 += aData[M * n + m] * bData[n + N * k];
                res6 += aData[M * (n + 1) + m] * bData[n + 1 + N * (k + 1)];
                res7 += aData[M * (n + 2) + m] * bData[n + 2 + N * (k + 2)];
                res8 += aData[M * (n + 3) + m] * bData[n + 3 + N * (k + 3)];

                res9 += aData[M * n + m] * bData[n + N * k];
                res10 += aData[M * (n + 1) + m] * bData[n + 1 + N * (k + 1)];
                res11 += aData[M * (n + 2) + m] * bData[n + 2 + N * (k + 2)];
                res12 += aData[M * (n + 3) + m] * bData[n + 3 + N * (k + 3)];

                res13 += aData[M * n + m] * bData[n + N * k];
                res14 += aData[M * (n + 1) + m] * bData[n + 1 + N * (k + 1)];
                res15 += aData[M * (n + 2) + m] * bData[n + 2 + N * (k + 2)];
                res16 += aData[M * (n + 3) + m] * bData[n + 3 + N * (k + 3)];
            }
            outData[M * k + m] = res1 + res2 + res3 + res4;
            outData[M * (k + 1) + m] = res5 + res6 + res7 + res8;
            outData[M * (k + 2) + m] = res9 + res10 + res11 + res12;
            outData[M * (k + 3) + m] = res13 + res14 + res15 + res16;
        }
    }
}

void naive_matmul_4x4_neon(Matrix &a, Matrix &b, Matrix &out)
{
    int M = a.rows;
    int N = b.rows;
    int K = b.cols;

    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

    for (int m = 0; m < M; m += 4)
    {
        for (int k = 0; k < K; k += 4)
        {
            float32x4_t res1 = {0, 0, 0, 0};
            float32x4_t res2 = {0, 0, 0, 0};
            float32x4_t res3 = {0, 0, 0, 0};
            float32x4_t res4 = {0, 0, 0, 0};

            float32x4_t va1, va2, va3, va4, vb1, vb2, vb3, vb4;
            for (int n = 0; n < N; n += 4)
            {
                va1 = vld1q_f32(aData + M * n + m);
                va2 = vld1q_f32(aData + M * (n + 1) + m);
                va3 = vld1q_f32(aData + M * (n + 2) + m);
                va4 = vld1q_f32(aData + M * (n + 3) + m);

                vb1 = vld1q_f32(bData + n + N * k);
                vb2 = vld1q_f32(bData + n + N * (k + 1));
                vb3 = vld1q_f32(bData + n + N * (k + 2));
                vb4 = vld1q_f32(bData + n + N * (k + 3));

                res1 = vfmaq_laneq_f32(res1, va1, vb1, 0);
                res2 = vfmaq_laneq_f32(res2, va1, vb2, 0);
                res3 = vfmaq_laneq_f32(res3, va1, vb3, 0);
                res4 = vfmaq_laneq_f32(res4, va1, vb4, 0);

                res1 = vfmaq_laneq_f32(res1, va2, vb1, 1);
                res2 = vfmaq_laneq_f32(res2, va2, vb2, 1);
                res3 = vfmaq_laneq_f32(res3, va2, vb3, 1);
                res4 = vfmaq_laneq_f32(res4, va2, vb4, 1);

                res1 = vfmaq_laneq_f32(res1, va3, vb1, 2);
                res2 = vfmaq_laneq_f32(res2, va3, vb2, 2);
                res3 = vfmaq_laneq_f32(res3, va3, vb3, 2);
                res4 = vfmaq_laneq_f32(res4, va3, vb4, 2);

                res1 = vfmaq_laneq_f32(res1, va4, vb1, 3);
                res2 = vfmaq_laneq_f32(res2, va4, vb2, 3);
                res3 = vfmaq_laneq_f32(res3, va4, vb3, 3);
                res4 = vfmaq_laneq_f32(res4, va4, vb4, 3);
            }

            vst1q_f32(outData + M * k + m, res1);
            vst1q_f32(outData + M * (k + 1) + m, res2);
            vst1q_f32(outData + M * (k + 2) + m, res3);
            vst1q_f32(outData + M * (k + 3) + m, res4);
        }
    }
}

void naive_matmul_4x4_neon_2(Matrix &a, Matrix &b_, Matrix &out)
{
    int M = a.rows;
    int N = a.cols;
    auto b = b_.transpose();
    int K = b.cols;

    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

    for (int m = 0; m < M - 3; m += 4)
    {
        for (int k = 0; k < K - 3; k += 4)
        {
            float32x4_t res1 = {0, 0, 0, 0};
            float32x4_t res2 = {0, 0, 0, 0};
            float32x4_t res3 = {0, 0, 0, 0};
            float32x4_t res4 = {0, 0, 0, 0};

            float32x4_t va1, vb1;
            for (int n = 0; n < N; n++)
            {
                va1 = vld1q_f32(aData + M * n + m);
                vb1 = vld1q_f32(bData + K * n + k);
                res1 = vfmaq_laneq_f32(res1, vb1, va1, 0);
                res2 = vfmaq_laneq_f32(res1, vb1, va1, 1);
                res3 = vfmaq_laneq_f32(res1, vb1, va1, 2);
                res4 = vfmaq_laneq_f32(res1, vb1, va1, 3);
            }

            vst1q_f32(outData + M * k + m, res1);
            vst1q_f32(outData + M * (k + 1) + m, res2);
            vst1q_f32(outData + M * (k + 2) + m, res3);
            vst1q_f32(outData + M * (k + 3) + m, res4);
        }
    }
}

void naive_matmul_4x4_neon_3(Matrix &a, Matrix &b_, Matrix &out)
{
    auto b = b_.transpose();
    int M = a.rows;
    int N = a.cols;
    int K = b.rows;

    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

    for (int m = 0; m < M - 3; m += 4)
    {
        for (int k = 0; k < K - 3; k += 4)
        {
            float32x4_t res1 = {0, 0, 0, 0};
            float32x4_t res2 = {0, 0, 0, 0};
            float32x4_t res3 = {0, 0, 0, 0};
            float32x4_t res4 = {0, 0, 0, 0};

            float32x4_t res5 = {0, 0, 0, 0};
            float32x4_t res6 = {0, 0, 0, 0};
            float32x4_t res7 = {0, 0, 0, 0};
            float32x4_t res8 = {0, 0, 0, 0};

            float32x4_t res9 = {0, 0, 0, 0};
            float32x4_t res10 = {0, 0, 0, 0};
            float32x4_t res11 = {0, 0, 0, 0};
            float32x4_t res12 = {0, 0, 0, 0};

            float32x4_t res13 = {0, 0, 0, 0};
            float32x4_t res14 = {0, 0, 0, 0};
            float32x4_t res15 = {0, 0, 0, 0};
            float32x4_t res16 = {0, 0, 0, 0};

            float32x4_t va1, va2, va3, va4, vb1, vb2, vb3, vb4;
            for (int n = 0; n < N-3; n += 4)
            {
                va1 = vld1q_f32(aData + M * n + m);
                vb1 = vld1q_f32(bData + K * n + k);
                res1 = vfmaq_laneq_f32(res1, vb1, va1, 0);
                res2 = vfmaq_laneq_f32(res2, vb1, va1, 1);
                res3 = vfmaq_laneq_f32(res3, vb1, va1, 2);
                res4 = vfmaq_laneq_f32(res4, vb1, va1, 3);

                va2 = vld1q_f32(aData + M * (n + 1) + m);
                vb2 = vld1q_f32(bData + K * (n + 1) + k);
                res5 = vfmaq_laneq_f32(res5, vb2, va2, 0);
                res6 = vfmaq_laneq_f32(res6, vb2, va2, 1);
                res7 = vfmaq_laneq_f32(res7, vb2, va2, 2);
                res8 = vfmaq_laneq_f32(res8, vb2, va2, 3);

                va3 = vld1q_f32(aData + M * (n + 2) + m);
                vb3 = vld1q_f32(bData + K * (n + 2) + k);
                res9 = vfmaq_laneq_f32(res9, vb3, va3, 0);
                res10 = vfmaq_laneq_f32(res10, vb3, va3, 1);
                res11 = vfmaq_laneq_f32(res11, vb3, va3, 2);
                res12 = vfmaq_laneq_f32(res12, vb3, va3, 3);

                va4 = vld1q_f32(aData + M * (n + 3) + m);
                vb4 = vld1q_f32(bData + K * (n + 3) + k);
                res13 = vfmaq_laneq_f32(res13, vb4, va4, 0);
                res14 = vfmaq_laneq_f32(res14, vb4, va4, 1);
                res15 = vfmaq_laneq_f32(res15, vb4, va4, 2);
                res16 = vfmaq_laneq_f32(res16, vb4, va4, 3);
            }

            res1 = vaddq_f32(res1, res5);
            res9 = vaddq_f32(res9, res13);
            res2 = vaddq_f32(res2, res6);
            res10 = vaddq_f32(res10, res14);
            res3 = vaddq_f32(res3, res7);
            res11 = vaddq_f32(res11, res15);
            res4 = vaddq_f32(res4, res8);
            res12 = vaddq_f32(res12, res16);
            res1 = vaddq_f32(res1, res9);
            res2 = vaddq_f32(res2, res10);
            res3 = vaddq_f32(res3, res11);
            res4 = vaddq_f32(res4, res12);

            vst1q_f32(outData + M * k + m, res1);
            vst1q_f32(outData + M * (k + 1) + m, res2);
            vst1q_f32(outData + M * (k + 2) + m, res3);
            vst1q_f32(outData + M * (k + 3) + m, res4);
        }
    }
}

void neon_matmul(Matrix &a, Matrix &b, Matrix &out)
{
    // auto a = a_.transpose();
    int N = a.rows;
    int M = a.cols;
    int K = b.cols;
    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();
    int n_step = 120;

    for (int n = 0; n < N - n_step + 1; n += n_step)
    {
        int n_ = fmin(n_step, N);
        for (int pk = 0; pk < K - 117; pk += 120)
        {
            for (int pm = 0; pm < M - 117; pm += 120)
            {
                inner_loop(n_, 120, 120, aData + pm, bData + pk, outData + pm * out.cols + pk, a.cols, b.cols, out.cols);
            }
        }
    }

    int remaining_M = M % 120;
    int remaining_K = K % 120;

    // // for (int m = 0; m < M - 11; m += 12)
    // // {
    // //     for (int k = 0; k < K - 7; k += 8)
    // //     {
    // //         matmul_12x8_micro_kernel_asmb(N, aData + m, bData + k, outData + m * out.cols + k, a.cols, b.cols, out.cols);
    // //     }
    // // }

    // // // for (int m = 0; m < M - 3; m += 4)
    // // // {
    // // //     for (int k = 0; k < K - 3; k += 4)
    // // //     {
    // // //         matmul_4x4_micro_kernel(N, aData + m, bData + k, outData + m * out.cols + k, a.cols, b.cols, out.cols);
    // // //     }
    // // // }

    // int remaining_M = M % 12;
    // int remaining_K = K % 8;

    for (size_t m = 0; m < M - 3; m += 4)
    {
        for (size_t k = K - remaining_K; k < K - 3; k += 4)
        {
            matmul_4x4_micro_kernel(N, aData + m, bData + k, outData + m * out.cols + k, a.cols, b.cols, out.cols);
        }
    }

    for (size_t m = M - remaining_M; m < M - 3; m += 4)
    {
        for (size_t k = 0; k < K - remaining_K - 3; k += 4)
        {
            matmul_4x4_micro_kernel(N, aData + m, bData + k, outData + m * out.cols + k, a.cols, b.cols, out.cols);
        }
    }
}

void inner_loop(int N, int M, int K, float *aData, float *bData, float *outData, int lda, int ldb, int ldc)
{
    // std::cout << N<<","<<M<<","<<K<<std::endl;
    for (int m = 0; m < M - 3; m += 4)
    {
        for (int k = 0; k < K - 3; k += 4)
        {
            matmul_4x4_micro_kernel(N, aData + m, bData + k, outData + m * ldc + k, lda, ldb, ldc);
        }
    }
}