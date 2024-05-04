#include "matmul.h"
#include <arm_neon.h>

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
    int K = b.rows;
    float *aData = a.data();
    float *bData = b.data();
    float *outData = out.data();

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
    float32x4_t res17 = {0, 0, 0, 0};
    float32x4_t res18 = {0, 0, 0, 0};
    float32x4_t res19 = {0, 0, 0, 0};
    float32x4_t res20 = {0, 0, 0, 0};
    float32x4_t res21 = {0, 0, 0, 0};
    float32x4_t res22 = {0, 0, 0, 0};
    float32x4_t res23 = {0, 0, 0, 0};
    float32x4_t res24 = {0, 0, 0, 0};

    float32x4_t va1 = vld1q_f32(aData);
    float32x4_t vb1_1 = vld1q_f32(bData);
    float32x4_t va2 = vld1q_f32(aData + 4);
    float32x4_t vb2_1 = vld1q_f32(bData + 4);
    float32x4_t vb1_2;
    float32x4_t vb2_2;
    float32x4_t va3 = vld1q_f32(aData + 8);

    aData += 12;
    bData += 8;

    for (size_t n = 0; n < 8; n++) // n
    {
        if (n % 2 == 0)
        {
            vb1_2 = vld1q_f32(bData);
            res1 = vfmaq_laneq_f32(res1, vb1_1, va1, 0);
            res2 = vfmaq_laneq_f32(res2, vb2_1, va1, 0); // Row1

            res3 = vfmaq_laneq_f32(res3, vb1_1, va1, 1);
            res4 = vfmaq_laneq_f32(res4, vb2_1, va1, 1); // Row2

            res5 = vfmaq_laneq_f32(res5, vb1_1, va1, 2);
            res6 = vfmaq_laneq_f32(res6, vb2_1, va1, 2); // Row3

            res7 = vfmaq_laneq_f32(res7, vb1_1, va1, 3);
            res8 = vfmaq_laneq_f32(res8, vb2_1, va1, 3); // Row4
            va1 = vld1q_f32(aData);
            vb2_2 = vld1q_f32(bData + 4);

            res9 = vfmaq_laneq_f32(res9, vb1_1, va2, 0);
            res10 = vfmaq_laneq_f32(res10, vb2_1, va2, 0); // Row8

            res11 = vfmaq_laneq_f32(res11, vb1_1, va2, 1);
            res12 = vfmaq_laneq_f32(res12, vb2_1, va2, 1); // Row9

            res13 = vfmaq_laneq_f32(res13, vb1_1, va2, 2);
            res14 = vfmaq_laneq_f32(res14, vb2_1, va2, 2); // Row10

            res15 = vfmaq_laneq_f32(res15, vb1_1, va2, 3);
            res16 = vfmaq_laneq_f32(res16, vb2_1, va2, 3); // Row11

            va2 = vld1q_f32(aData + 4);

            res17 = vfmaq_laneq_f32(res17, vb1_1, va3, 0);
            res18 = vfmaq_laneq_f32(res18, vb2_1, va3, 0); // Row12

            res19 = vfmaq_laneq_f32(res19, vb1_1, va3, 1);
            res20 = vfmaq_laneq_f32(res20, vb2_1, va3, 1); // Row13

            res21 = vfmaq_laneq_f32(res21, vb1_1, va3, 2);
            res22 = vfmaq_laneq_f32(res22, vb2_1, va3, 2); // Row14

            res23 = vfmaq_laneq_f32(res23, vb1_1, va3, 3);
            res24 = vfmaq_laneq_f32(res24, vb2_1, va3, 3); // Row15
            va3 = vld1q_f32(aData + 8);
        }
        else
        {
            vb1_1 = vld1q_f32(bData);
            res1 = vfmaq_laneq_f32(res1, vb1_2, va1, 0);
            res2 = vfmaq_laneq_f32(res2, vb2_2, va1, 0); // Row1

            res3 = vfmaq_laneq_f32(res3, vb1_2, va1, 1);
            res4 = vfmaq_laneq_f32(res4, vb2_2, va1, 1); // Row2

            res5 = vfmaq_laneq_f32(res5, vb1_2, va1, 2);
            res6 = vfmaq_laneq_f32(res6, vb2_2, va1, 2); // Row3

            res7 = vfmaq_laneq_f32(res7, vb1_2, va1, 3);
            res8 = vfmaq_laneq_f32(res8, vb2_2, va1, 3); // Row4
            va1 = vld1q_f32(aData);
            vb2_1 = vld1q_f32(bData);

            res9 = vfmaq_laneq_f32(res9, vb1_2, va2, 0);
            res10 = vfmaq_laneq_f32(res10, vb2_2, va2, 0); // Row8

            res11 = vfmaq_laneq_f32(res11, vb1_2, va2, 1);
            res12 = vfmaq_laneq_f32(res12, vb2_2, va2, 1); // Row9

            res13 = vfmaq_laneq_f32(res13, vb1_2, va2, 2);
            res14 = vfmaq_laneq_f32(res14, vb2_2, va2, 2); // Row10

            res15 = vfmaq_laneq_f32(res15, vb1_2, va2, 3);
            res16 = vfmaq_laneq_f32(res16, vb2_2, va2, 3); // Row11
            va2 = vld1q_f32(aData + 4);

            res17 = vfmaq_laneq_f32(res17, vb1_2, va3, 0);
            res18 = vfmaq_laneq_f32(res18, vb2_2, va3, 0); // Row12

            res19 = vfmaq_laneq_f32(res19, vb1_2, va3, 1);
            res20 = vfmaq_laneq_f32(res20, vb2_2, va3, 1); // Row13

            res21 = vfmaq_laneq_f32(res21, vb1_2, va3, 2);
            res22 = vfmaq_laneq_f32(res22, vb2_2, va3, 2); // Row14

            res23 = vfmaq_laneq_f32(res23, vb1_2, va3, 3);
            res24 = vfmaq_laneq_f32(res24, vb2_2, va3, 3); // Row15
            va3 = vld1q_f32(aData + 8);
        }
        aData += 12;
        bData += 8;
    }

    vst1q_f32(outData, res1);
    vst1q_f32(outData + 4, res2);
    vst1q_f32(outData + 8, res3);
    vst1q_f32(outData + 12, res4);
    vst1q_f32(outData + 16, res5);
    vst1q_f32(outData + 20, res6);
    vst1q_f32(outData + 24, res7);
    vst1q_f32(outData + 28, res8);
    vst1q_f32(outData + 32, res9);
    vst1q_f32(outData + 36, res10);
    vst1q_f32(outData + 40, res11);
    vst1q_f32(outData + 44, res12);
    vst1q_f32(outData + 48, res13);
    vst1q_f32(outData + 52, res14);
    vst1q_f32(outData + 56, res15);
    vst1q_f32(outData + 60, res16);
    vst1q_f32(outData + 64, res17);
    vst1q_f32(outData + 68, res18);
    vst1q_f32(outData + 72, res19);
    vst1q_f32(outData + 76, res20);
    vst1q_f32(outData + 80, res21);
    vst1q_f32(outData + 84, res22);
    vst1q_f32(outData + 88, res23);
    vst1q_f32(outData + 92, res24);
}