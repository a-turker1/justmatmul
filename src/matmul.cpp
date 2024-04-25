#include "matmul.h"
#include <arm_neon.h>

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


void neon_matmul(Matrix &a, Matrix &b, Matrix &out)
{
    int M = a.cols;
    int N = a.rows;
    int K = b.rows;
    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k ++)
        {
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            float32x4_t sum4 = vdupq_n_f32(0.0f);
            for (int n = 0; n < N; n+=16)
            {
                float32x4_t va1 = vld1q_f32(aData + m * N + n);
                float32x4_t vb1 = vld1q_f32(bData + n * K + k);
                float32x4_t va2 = vld1q_f32(aData + m * N + (n +4));
                float32x4_t vb2 = vld1q_f32(bData + (n +4) * K + k);
                float32x4_t va3 = vld1q_f32(aData + m * N + (n+8));
                float32x4_t vb3 = vld1q_f32(bData + (n+8) * K + k);
                float32x4_t va4 = vld1q_f32(aData + m * N + (n+12));
                float32x4_t vb4 = vld1q_f32(bData + (n+12) * K + k);
                sum1 = vmlaq_f32(sum1, va1, vb1);
                sum2 = vmlaq_f32(sum2, va2, vb2);
                sum3 = vmlaq_f32(sum3, va3, vb3);
                sum4 = vmlaq_f32(sum4, va4, vb4);
            }
            sum1 = vaddq_f32(sum1, sum2);
            sum2 = vaddq_f32(sum4, sum3);
            sum1 = vaddq_f32(sum1, sum2);
            float32x2_t sum_high = vadd_f32(vget_high_f32(sum1), vget_low_f32(sum1));
            float32x2_t sum_low = vpadd_f32(sum_high, sum_high);
            outData[m * M + k] = vget_lane_f32(sum_low, 0);
        }
    }
}