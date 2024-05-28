#include <arm_neon.h>

#define STORE_RESULTS(index, res1, res2)        \
    vst1q_f32(outData + (index) * s_out, res1); \
    vst1q_f32(outData + (index) * s_out + 4, res2);

#define vfloat4 float32x4_t

/*
Applies operation A@B=OUT.
Expected A (mxn) transposed, B(NXK) row major and out (MXK) row major.
*/

void naive_matmul_col_major_kernel(int M, int N, int K, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out)
{
    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k++)
        {
            float res = 0;
            for (int n = 0; n < N; n++)
            {
                res += aData[s_a * n + m] * bData[n * s_b + k];
            }
            outData[s_out * k + m] = res;
        }
    }
}

void naive_matmul_row_major_kernel(int M, int N, int K, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out)
{
    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k++)
        {
            float res = 0;
            for (int n = 0; n < N; n++)
            {
                res += aData[s_a * n + m] * bData[n * s_b + k];
            }
            outData[s_out * m + k] = res;
        }
    }
}

void matmul_12x8_micro_kernel_row_major(int N, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out)
{
    vfloat4 res1 = {0, 0, 0, 0};
    vfloat4 res2 = {0, 0, 0, 0};
    vfloat4 res3 = {0, 0, 0, 0};
    vfloat4 res4 = {0, 0, 0, 0};

    vfloat4 res5 = {0, 0, 0, 0};
    vfloat4 res6 = {0, 0, 0, 0};
    vfloat4 res7 = {0, 0, 0, 0};
    vfloat4 res8 = {0, 0, 0, 0};

    vfloat4 res9 = {0, 0, 0, 0};
    vfloat4 res10 = {0, 0, 0, 0};
    vfloat4 res11 = {0, 0, 0, 0};
    vfloat4 res12 = {0, 0, 0, 0};

    vfloat4 res13 = {0, 0, 0, 0};
    vfloat4 res14 = {0, 0, 0, 0};
    vfloat4 res15 = {0, 0, 0, 0};
    vfloat4 res16 = {0, 0, 0, 0};

    vfloat4 res17 = {0, 0, 0, 0};
    vfloat4 res18 = {0, 0, 0, 0};
    vfloat4 res19 = {0, 0, 0, 0};
    vfloat4 res20 = {0, 0, 0, 0};

    vfloat4 res21 = {0, 0, 0, 0};
    vfloat4 res22 = {0, 0, 0, 0};
    vfloat4 res23 = {0, 0, 0, 0};
    vfloat4 res24 = {0, 0, 0, 0};

    vfloat4 va1 = vld1q_f32(aData);
    vfloat4 vb1 = vld1q_f32(bData);
    vfloat4 va2 = vld1q_f32(aData + 4);
    vfloat4 vb2 = vld1q_f32(bData + 4);
    vfloat4 va3 = vld1q_f32(aData + 8);

    aData += s_a;
    bData += s_b;

    for (size_t n = 0; n < N; n++) // n
    {
        res1 = vfmaq_laneq_f32(res1, vb1, va1, 0);
        res2 = vfmaq_laneq_f32(res2, vb2, va1, 0); // Row1

        res3 = vfmaq_laneq_f32(res3, vb1, va1, 1);
        res4 = vfmaq_laneq_f32(res4, vb2, va1, 1); // Row2

        res5 = vfmaq_laneq_f32(res5, vb1, va1, 2);
        res6 = vfmaq_laneq_f32(res6, vb2, va1, 2); // Row3

        res7 = vfmaq_laneq_f32(res7, vb1, va1, 3);
        res8 = vfmaq_laneq_f32(res8, vb2, va1, 3); // Row4
        va1 = vld1q_f32(aData);

        res9 = vfmaq_laneq_f32(res9, vb1, va2, 0);
        res10 = vfmaq_laneq_f32(res10, vb2, va2, 0); // Row8

        res11 = vfmaq_laneq_f32(res11, vb1, va2, 1);
        res12 = vfmaq_laneq_f32(res12, vb2, va2, 1); // Row9

        res13 = vfmaq_laneq_f32(res13, vb1, va2, 2);
        res14 = vfmaq_laneq_f32(res14, vb2, va2, 2); // Row10

        res15 = vfmaq_laneq_f32(res15, vb1, va2, 3);
        res16 = vfmaq_laneq_f32(res16, vb2, va2, 3); // Row11
        va2 = vld1q_f32(aData + 4);

        res17 = vfmaq_laneq_f32(res17, vb1, va3, 0);
        res18 = vfmaq_laneq_f32(res18, vb2, va3, 0); // Row12

        res19 = vfmaq_laneq_f32(res19, vb1, va3, 1);
        res20 = vfmaq_laneq_f32(res20, vb2, va3, 1); // Row13

        res21 = vfmaq_laneq_f32(res21, vb1, va3, 2);
        res22 = vfmaq_laneq_f32(res22, vb2, va3, 2); // Row14

        res23 = vfmaq_laneq_f32(res23, vb1, va3, 3);
        res24 = vfmaq_laneq_f32(res24, vb2, va3, 3); // Row15
        vb1 = vld1q_f32(bData);
        vb2 = vld1q_f32(bData + 4);
        va3 = vld1q_f32(aData + 8);

        aData += s_a;
        bData += s_b;
    }

    vst1q_f32(outData, res1);
    vst1q_f32(outData + 4, res2);
    vst1q_f32(outData + s_out, res3);
    vst1q_f32(outData + s_out + 4, res4);
    vst1q_f32(outData + 2 * s_out, res5);
    vst1q_f32(outData + 2 * s_out + 4, res6);
    vst1q_f32(outData + 3 * s_out, res7);
    vst1q_f32(outData + 3 * s_out + 4, res8);
    vst1q_f32(outData + 4 * s_out, res9);
    vst1q_f32(outData + 4 * s_out + 4, res10);
    vst1q_f32(outData + 5 * s_out, res11);
    vst1q_f32(outData + 5 * s_out + 4, res12);
    vst1q_f32(outData + 6 * s_out, res13);
    vst1q_f32(outData + 6 * s_out + 4, res14);
    vst1q_f32(outData + 7 * s_out, res15);
    vst1q_f32(outData + 7 * s_out + 4, res16);
    vst1q_f32(outData + 8 * s_out, res17);
    vst1q_f32(outData + 8 * s_out + 4, res18);
    vst1q_f32(outData + 9 * s_out, res19);
    vst1q_f32(outData + 9 * s_out + 4, res20);
    vst1q_f32(outData + 10 * s_out, res21);
    vst1q_f32(outData + 10 * s_out + 4, res22);
    vst1q_f32(outData + 11 * s_out, res23);
    vst1q_f32(outData + 11 * s_out + 4, res24);
}


void matmul_4x4_micro_kernel_row_major(int N, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out)
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
    for (int n = 0; n < N - 3; n += 4)
    {
        va1 = vld1q_f32(aData + s_a * n);
        vb1 = vld1q_f32(bData + s_b * n);
        res1 = vfmaq_laneq_f32(res1, vb1, va1, 0);
        res2 = vfmaq_laneq_f32(res2, vb1, va1, 1);
        res3 = vfmaq_laneq_f32(res3, vb1, va1, 2);
        res4 = vfmaq_laneq_f32(res4, vb1, va1, 3);

        va2 = vld1q_f32(aData + s_a * (n + 1));
        vb2 = vld1q_f32(bData + s_b * (n + 1));
        res5 = vfmaq_laneq_f32(res5, vb2, va2, 0);
        res6 = vfmaq_laneq_f32(res6, vb2, va2, 1);
        res7 = vfmaq_laneq_f32(res7, vb2, va2, 2);
        res8 = vfmaq_laneq_f32(res8, vb2, va2, 3);

        va3 = vld1q_f32(aData + s_a * (n + 2));
        vb3 = vld1q_f32(bData + s_b * (n + 2));
        res9 = vfmaq_laneq_f32(res9, vb3, va3, 0);
        res10 = vfmaq_laneq_f32(res10, vb3, va3, 1);
        res11 = vfmaq_laneq_f32(res11, vb3, va3, 2);
        res12 = vfmaq_laneq_f32(res12, vb3, va3, 3);

        va4 = vld1q_f32(aData + s_a * (n + 3));
        vb4 = vld1q_f32(bData + s_b * (n + 3));
        res13 = vfmaq_laneq_f32(res13, vb4, va4, 0);
        res14 = vfmaq_laneq_f32(res14, vb4, va4, 1);
        res15 = vfmaq_laneq_f32(res15, vb4, va4, 2);
        res16 = vfmaq_laneq_f32(res16, vb4, va4, 3);
    }

    for (int n = N - N % 4; n < N; n++)
    {
        va1 = vld1q_f32(aData + s_a * n);
        vb1 = vld1q_f32(bData + s_b * n);
        res1 = vfmaq_laneq_f32(res1, vb1, va1, 0);
        res2 = vfmaq_laneq_f32(res2, vb1, va1, 1);
        res3 = vfmaq_laneq_f32(res3, vb1, va1, 2);
        res4 = vfmaq_laneq_f32(res4, vb1, va1, 3);
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

    vst1q_f32(outData, res1);
    vst1q_f32(outData + s_out, res2);
    vst1q_f32(outData + 2 * s_out, res3);
    vst1q_f32(outData + 3 * s_out, res4);
}


void matmul_4x4_micro_kernel_col_major(int N, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out)
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
    for (int n = 0; n < N - 3; n += 4)
    {
        va1 = vld1q_f32(aData + s_a * n);
        vb1 = vld1q_f32(bData + s_b * n);
        res1 = vfmaq_laneq_f32(res1, va1, vb1, 0);
        res2 = vfmaq_laneq_f32(res2, va1, vb1, 1);
        res3 = vfmaq_laneq_f32(res3, va1, vb1, 2);
        res4 = vfmaq_laneq_f32(res4, va1, vb1, 3);

        va2 = vld1q_f32(aData + s_a * (n + 1));
        vb2 = vld1q_f32(bData + s_b * (n + 1));
        res5 = vfmaq_laneq_f32(res5, va2, vb2, 0);
        res6 = vfmaq_laneq_f32(res6, va2, vb2, 1);
        res7 = vfmaq_laneq_f32(res7, va2, vb2, 2);
        res8 = vfmaq_laneq_f32(res8, va2, vb2, 3);

        va3 = vld1q_f32(aData + s_a * (n + 2));
        vb3 = vld1q_f32(bData + s_b * (n + 2));
        res9 = vfmaq_laneq_f32(res9, va3, vb3, 0);
        res10 = vfmaq_laneq_f32(res10, va3, vb3, 1);
        res11 = vfmaq_laneq_f32(res11, va3, vb3, 2);
        res12 = vfmaq_laneq_f32(res12, va3, vb3, 3);

        va4 = vld1q_f32(aData + s_a * (n + 3));
        vb4 = vld1q_f32(bData + s_b * (n + 3));
        res13 = vfmaq_laneq_f32(res13, va4, vb4, 0);
        res14 = vfmaq_laneq_f32(res14, va4, vb4, 1);
        res15 = vfmaq_laneq_f32(res15, va4, vb4, 2);
        res16 = vfmaq_laneq_f32(res16, va4, vb4, 3);
    }

    for (int n = N - N % 4; n < N; n++)
    {
        va1 = vld1q_f32(aData + s_a * n);
        vb1 = vld1q_f32(bData + s_b * n);
        res1 = vfmaq_laneq_f32(res1, vb1, va1, 0);
        res2 = vfmaq_laneq_f32(res2, vb1, va1, 1);
        res3 = vfmaq_laneq_f32(res3, vb1, va1, 2);
        res4 = vfmaq_laneq_f32(res4, vb1, va1, 3);
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

    vst1q_f32(outData, res1);
    vst1q_f32(outData + s_out, res2);
    vst1q_f32(outData + 2 * s_out, res3);
    vst1q_f32(outData + 3 * s_out, res4);
}

/*
Applies opreation A@B.T=OUT
Expected A (mxn), B(NXK) column major transposed and out (MXK) row major.
*/
void matmul_12x8_micro_kernel_col_major(int N, float *aData, float *bData, float *outData, int lda, int ldb, int ldout)
{
    vfloat4 res1 = vld1q_f32(outData);
    vfloat4 res2 = vld1q_f32(outData + 4);
    vfloat4 res3 = vld1q_f32(outData + 8);

    vfloat4 res4 = vld1q_f32(outData + ldout);
    vfloat4 res5 = vld1q_f32(outData + ldout + 4);
    vfloat4 res6 = vld1q_f32(outData + ldout + 8);

    vfloat4 res7 = vld1q_f32(outData + 2 * ldout);
    vfloat4 res8 = vld1q_f32(outData + 2 * ldout + 4);
    vfloat4 res9 = vld1q_f32(outData + 2 * ldout + 8);

    vfloat4 res10 = vld1q_f32(outData + 3 * ldout);
    vfloat4 res11 = vld1q_f32(outData + 3 * ldout + 4);
    vfloat4 res12 = vld1q_f32(outData + 3 * ldout + 8);

    vfloat4 res13 = vld1q_f32(outData + 4 * ldout);
    vfloat4 res14 = vld1q_f32(outData + 4 * ldout + 4);
    vfloat4 res15 = vld1q_f32(outData + 4 * ldout + 8);

    vfloat4 res16 = vld1q_f32(outData + 5 * ldout);
    vfloat4 res17 = vld1q_f32(outData + 5 * ldout + 4);
    vfloat4 res18 = vld1q_f32(outData + 5 * ldout + 8);

    vfloat4 res19 = vld1q_f32(outData + 6 * ldout);
    vfloat4 res20 = vld1q_f32(outData + 6 * ldout + 4);
    vfloat4 res21 = vld1q_f32(outData + 6 * ldout + 8);

    vfloat4 res22 = vld1q_f32(outData + 7 * ldout);
    vfloat4 res23 = vld1q_f32(outData + 7 * ldout + 4);
    vfloat4 res24 = vld1q_f32(outData + 7 * ldout + 8);

    vfloat4 va1 = vld1q_f32(aData);
    vfloat4 va2 = vld1q_f32(aData + 4);
    vfloat4 va3 = vld1q_f32(aData + 8);
    vfloat4 vb1 = vld1q_f32(bData);
    vfloat4 vb2 = vld1q_f32(bData + 4);

    aData += lda;
    bData += ldb;

    for (size_t n = 0; n < N; n++) // n
    {
        res1 = vfmaq_laneq_f32(res1, va1, vb1, 0);
        res2 = vfmaq_laneq_f32(res2, va2, vb1, 0);
        res3 = vfmaq_laneq_f32(res3, va3, vb1, 0); // Col1

        res4 = vfmaq_laneq_f32(res4, va1, vb1, 1);
        res5 = vfmaq_laneq_f32(res6, va2, vb1, 1);
        res6 = vfmaq_laneq_f32(res7, va3, vb1, 1); // Col2

        res7 = vfmaq_laneq_f32(res7, va1, vb1, 2);
        res8 = vfmaq_laneq_f32(res8, va2, vb1, 2);
        res9 = vfmaq_laneq_f32(res9, va3, vb1, 2); // Col3

        res10 = vfmaq_laneq_f32(res10, va1, vb1, 3);
        res11 = vfmaq_laneq_f32(res11, va2, vb1, 3);
        res12 = vfmaq_laneq_f32(res12, va3, vb1, 3); // Col4
        vb1 = vld1q_f32(bData);

        res13 = vfmaq_laneq_f32(res13, va1, vb2, 0);
        res14 = vfmaq_laneq_f32(res14, va2, vb2, 0);
        res15 = vfmaq_laneq_f32(res15, va3, vb2, 0); // Col5

        res16 = vfmaq_laneq_f32(res16, va1, vb2, 1);
        res17 = vfmaq_laneq_f32(res17, va2, vb2, 1);
        res18 = vfmaq_laneq_f32(res18, va3, vb2, 1); // Col6

        res19 = vfmaq_laneq_f32(res19, va1, vb2, 2);
        res20 = vfmaq_laneq_f32(res20, va2, vb2, 2);
        res21 = vfmaq_laneq_f32(res21, va3, vb2, 2); // Col7

        res22 = vfmaq_laneq_f32(res22, va1, vb2, 3);
        res23 = vfmaq_laneq_f32(res23, va2, vb2, 3);
        res24 = vfmaq_laneq_f32(res24, va3, vb2, 3); // Col8

        va1 = vld1q_f32(aData);
        va2 = vld1q_f32(aData + 4);
        va3 = vld1q_f32(aData + 8);
        vb2 = vld1q_f32(bData + 4);

        aData += lda;
        bData += ldb;
    }

    vst1q_f32(outData, res1);
    vst1q_f32(outData + 4, res2);
    vst1q_f32(outData + 8, res3);
    vst1q_f32(outData + ldout, res4);
    vst1q_f32(outData + ldout + 4, res5);
    vst1q_f32(outData + ldout + 8, res6);
    vst1q_f32(outData + 2 * ldout, res7);
    vst1q_f32(outData + 2 * ldout + 4, res8);
    vst1q_f32(outData + 2 * ldout + 8, res9);
    vst1q_f32(outData + 3 * ldout, res10);
    vst1q_f32(outData + 3 * ldout + 4, res11);
    vst1q_f32(outData + 3 * ldout + 8, res12);
    vst1q_f32(outData + 4 * ldout, res13);
    vst1q_f32(outData + 4 * ldout + 4, res14);
    vst1q_f32(outData + 4 * ldout + 8, res15);
    vst1q_f32(outData + 5 * ldout, res16);
    vst1q_f32(outData + 5 * ldout + 4, res17);
    vst1q_f32(outData + 5 * ldout + 8, res18);
    vst1q_f32(outData + 6 * ldout, res19);
    vst1q_f32(outData + 6 * ldout + 4, res20);
    vst1q_f32(outData + 6 * ldout + 8, res21);
    vst1q_f32(outData + 7 * ldout, res22);
    vst1q_f32(outData + 7 * ldout + 4, res23);
    vst1q_f32(outData + 7 * ldout + 8, res24);
}

void matmul_12x8_micro_kernel_asmb(int N, float *aData, float *bData, float *outData, long s_a, long s_b, long s_out)
{
    s_a *= 4;
    s_b *= 4;
    s_out *= 4;
    __asm__ volatile(
        ""
        "dup v7.4s, wzr                 \n\t"
        "ldr x0, %[aaddr]               \n\t"
        "dup v8.4s, wzr                 \n\t"
        "ldr x1, %[baddr]               \n\t"
        "dup v9.4s, wzr                 \n\t"
        "ldr x2, %[outaddr]             \n\t"
        "dup v10.4s, wzr                \n\t"
        "ldr x3, %[n]                  \n\t"
        "dup v11.4s, wzr                \n\t"
        "ldr x4, %[s_a]                  \n\t"
        "dup v12.4s, wzr                \n\t"
        "ldr x5, %[s_b]                  \n\t"
        "dup v13.4s, wzr                \n\t"
        "ldr x6, %[s_out]                  \n\t"
        "dup v14.4s, wzr                \n\t"
        "ldp q0, q1, [x0]               \n\t"
        "dup v15.4s, wzr                \n\t"
        "ldr q2, [x0, #32]              \n\t"
        "dup v16.4s, wzr                \n\t"
        "ldp q3, q4, [x1]               \n\t"
        "dup v17.4s, wzr                \n\t"
        "add x0, x0, x4                \n\t"
        "dup v18.4s, wzr                \n\t"
        "add x1, x1, x5                \n\t"
        "dup v19.4s, wzr                \n\t"
        "dup v20.4s, wzr                \n\t"
        "dup v21.4s, wzr                \n\t"
        "dup v22.4s, wzr                \n\t"
        "dup v23.4s, wzr                \n\t"
        "dup v23.4s, wzr                \n\t"
        "dup v24.4s, wzr                \n\t"
        "dup v25.4s, wzr                \n\t"
        "dup v26.4s, wzr                \n\t"
        "dup v27.4s, wzr                \n\t"
        "dup v28.4s, wzr                \n\t"
        "dup v29.4s, wzr                \n\t"
        "dup v30.4s, wzr                \n\t"
        "                               \n\t"
        "main_iter%=:                     \n\t"
        "ldp q5, q6, [x1]                \n\t"
        "add x1, x1, x5                \n\t"
        "fmla.4s v7, v3, v0[0]          \n\t"
        "fmla.4s v8, v4, v0[0]          \n\t"
        "fmla.4s v9, v3, v0[1]          \n\t"
        "fmla.4s v10, v4, v0[1]         \n\t"
        "fmla.4s v11, v3, v0[2]         \n\t"
        "fmla.4s v12, v4, v0[2]         \n\t"
        "fmla.4s v13, v3, v0[3]         \n\t"
        "fmla.4s v14, v4, v0[3]         \n\t"
        "ldr q0, [x0]                   \n\t"
        "fmla.4s v15, v3, v1[0]          \n\t"
        "fmla.4s v16, v4, v1[0]          \n\t"
        "fmla.4s v17, v3, v1[1]          \n\t"
        "fmla.4s v18, v4, v1[1]         \n\t"
        "fmla.4s v19, v3, v1[2]         \n\t"
        "fmla.4s v20, v4, v1[2]         \n\t"
        "fmla.4s v21, v3, v1[3]         \n\t"
        "fmla.4s v22, v4, v1[3]         \n\t"
        "ldr q1, [x0, #16]              \n\t"
        "fmla.4s v23, v3, v2[0]          \n\t"
        "fmla.4s v24, v4, v2[0]          \n\t"
        "fmla.4s v25, v3, v2[1]          \n\t"
        "fmla.4s v26, v4, v2[1]         \n\t"
        "fmla.4s v27, v3, v2[2]         \n\t"
        "fmla.4s v28, v4, v2[2]         \n\t"
        "fmla.4s v29, v3, v2[3]         \n\t"
        "fmla.4s v30, v4, v2[3]         \n\t" // Iter 0
        "ldr q2, [x0, #32]              \n\t"
        "add x0, x0, x4                 \n\t"
        "fmla.4s v7, v5, v0[0]          \n\t"
        "fmla.4s v8, v6, v0[0]          \n\t"
        "fmla.4s v9, v5, v0[1]          \n\t"
        "ldp q3, q4, [x1]          \n\t"
        "add x1, x1, x5                \n\t"
        "fmla.4s v10, v6, v0[1]         \n\t"
        "fmla.4s v11, v5, v0[2]         \n\t"
        "fmla.4s v12, v6, v0[2]         \n\t"
        "fmla.4s v13, v5, v0[3]         \n\t"
        "fmla.4s v14, v6, v0[3]         \n\t"
        "ldr q0, [x0]                   \n\t"
        "fmla.4s v15, v5, v1[0]          \n\t"
        "fmla.4s v16, v6, v1[0]          \n\t"
        "fmla.4s v17, v5, v1[1]          \n\t"
        "fmla.4s v18, v6, v1[1]         \n\t"
        "fmla.4s v19, v5, v1[2]         \n\t"
        "fmla.4s v20, v6, v1[2]         \n\t"
        "fmla.4s v21, v5, v1[3]         \n\t"
        "fmla.4s v22, v6, v1[3]         \n\t"
        "ldr q1, [x0, #16]              \n\t"
        "fmla.4s v23, v5, v2[0]          \n\t"
        "fmla.4s v24, v6, v2[0]          \n\t"
        "fmla.4s v25, v5, v2[1]          \n\t"
        "fmla.4s v26, v6, v2[1]         \n\t"
        "fmla.4s v27, v5, v2[2]         \n\t"
        "fmla.4s v28, v6, v2[2]         \n\t"
        "fmla.4s v29, v5, v2[3]         \n\t"
        "fmla.4s v30, v6, v2[3]         \n\t" // Iter 1
        "ldr q2, [x0, #32]              \n\t"
        "add x0, x0, x4                \n\t"
        "sub x3, x3, 2                  \n\t"
        "cmp x3, 2                       \n\t"
        "bne main_iter%=                  \n\t"

        "ldp q5, q6, [x1]          \n\t"
        "fmla.4s v7, v3, v0[0]          \n\t" // Last iter
        "fmla.4s v8, v4, v0[0]          \n\t"
        "fmla.4s v9, v3, v0[1]          \n\t"
        "fmla.4s v10, v4, v0[1]         \n\t"
        "fmla.4s v11, v3, v0[2]         \n\t"
        "fmla.4s v12, v4, v0[2]         \n\t"
        "fmla.4s v13, v3, v0[3]         \n\t"
        "fmla.4s v14, v4, v0[3]         \n\t"
        "ldr q0, [x0]                   \n\t"
        "fmla.4s v15, v3, v1[0]          \n\t"
        "fmla.4s v16, v4, v1[0]          \n\t"
        "fmla.4s v17, v3, v1[1]          \n\t"
        "fmla.4s v18, v4, v1[1]         \n\t"
        "fmla.4s v19, v3, v1[2]         \n\t"
        "fmla.4s v20, v4, v1[2]         \n\t"
        "fmla.4s v21, v3, v1[3]         \n\t"
        "fmla.4s v22, v4, v1[3]         \n\t"
        "ldr q1, [x0, #16]              \n\t"
        "fmla.4s v23, v3, v2[0]          \n\t"
        "fmla.4s v24, v4, v2[0]          \n\t"
        "fmla.4s v25, v3, v2[1]          \n\t"
        "fmla.4s v26, v4, v2[1]         \n\t"
        "fmla.4s v27, v3, v2[2]         \n\t"
        "fmla.4s v28, v4, v2[2]         \n\t"
        "fmla.4s v29, v3, v2[3]         \n\t"
        "fmla.4s v30, v4, v2[3]         \n\t" // Iter 0
        "ldr q2, [x0, #32]              \n\t"

        "fmla.4s v7, v5, v0[0]          \n\t"
        "fmla.4s v8, v6, v0[0]          \n\t"
        "fmla.4s v9, v5, v0[1]          \n\t"
        "fmla.4s v10, v6, v0[1]         \n\t"
        "fmla.4s v11, v5, v0[2]         \n\t"
        "fmla.4s v12, v6, v0[2]         \n\t"
        "fmla.4s v13, v5, v0[3]         \n\t"
        "fmla.4s v14, v6, v0[3]         \n\t"
        "fmla.4s v15, v5, v1[0]          \n\t"
        "fmla.4s v16, v6, v1[0]          \n\t"
        "fmla.4s v17, v5, v1[1]          \n\t"
        "fmla.4s v18, v6, v1[1]         \n\t"
        "fmla.4s v19, v5, v1[2]         \n\t"
        "fmla.4s v20, v6, v1[2]         \n\t"
        "fmla.4s v21, v5, v1[3]         \n\t"
        "fmla.4s v22, v6, v1[3]         \n\t"
        "fmla.4s v23, v5, v2[0]          \n\t"
        "fmla.4s v24, v6, v2[0]          \n\t"
        "fmla.4s v25, v5, v2[1]          \n\t"
        "fmla.4s v26, v6, v2[1]         \n\t"
        "fmla.4s v27, v5, v2[2]         \n\t"
        "fmla.4s v28, v6, v2[2]         \n\t"
        "fmla.4s v29, v5, v2[3]         \n\t"
        "fmla.4s v30, v6, v2[3]         \n\t" // Iter 1

        "stp q7, q8, [x2] \n\t"
        "add x2, x2, x6 \n\t"
        "stp q9, q10, [x2] \n\t"
        "add x2, x2, x6 \n\t"
        "stp q11, q12, [x2] \n\t"
        "add x2, x2, x6 \n\t"
        "stp q13, q14, [x2] \n\t"
        "add x2, x2, x6 \n\t"
        "stp q15, q16, [x2] \n\t"
        "add x2, x2, x6 \n\t"
        "stp q17, q18, [x2] \n\t"
        "add x2, x2, x6 \n\t"
        "stp Q19, q20, [x2] \n\t"
        "add x2, x2, x6 \n\t"
        "stp Q21, q22, [x2] \n\t"
        "add x2, x2, x6 \n\t"
        "stp Q23, q24, [x2] \n\t"
        "add x2, x2, x6 \n\t"
        "stp Q25, q26, [x2] \n\t"
        "add x2, x2, x6 \n\t"
        "stp Q27, q28, [x2] \n\t"
        "add x2, x2, x6 \n\t"
        "stp Q29, q30, [x2] \n\t"
        :
        :
        [aaddr] "m"(aData),
        [baddr] "m"(bData),
        [outaddr] "m"(outData),
        [n] "m"(N),
        [s_a] "m"(s_a),
        [s_b] "m"(s_b),
        [s_out] "m"(s_out)
        : "x0", "x1", "x2",
          "x3", "x4", "x5",
          "x6",
          "v0", "v1", "v2", "v3",
          "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27",
          "v28", "v29", "v30", "v31");
}

void matmul_4x4_micro_kernel(int N, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out)
{

    vfloat4 res1 = {0, 0, 0, 0};
    vfloat4 res2 = {0, 0, 0, 0};
    vfloat4 res3 = {0, 0, 0, 0};
    vfloat4 res4 = {0, 0, 0, 0};

    vfloat4 res5 = {0, 0, 0, 0};
    vfloat4 res6 = {0, 0, 0, 0};
    vfloat4 res7 = {0, 0, 0, 0};
    vfloat4 res8 = {0, 0, 0, 0};

    vfloat4 res9 = {0, 0, 0, 0};
    vfloat4 res10 = {0, 0, 0, 0};
    vfloat4 res11 = {0, 0, 0, 0};
    vfloat4 res12 = {0, 0, 0, 0};

    vfloat4 res13 = {0, 0, 0, 0};
    vfloat4 res14 = {0, 0, 0, 0};
    vfloat4 res15 = {0, 0, 0, 0};
    vfloat4 res16 = {0, 0, 0, 0};

    vfloat4 va1 = vld1q_f32(aData);
    vfloat4 va2 = vld1q_f32(aData + s_a);
    vfloat4 va3 = vld1q_f32(aData + 2 * s_a);
    vfloat4 va4;

    vfloat4 vb1 = vld1q_f32(bData);
    vfloat4 vb2 = vld1q_f32(bData + s_b);
    vfloat4 vb3 = vld1q_f32(bData + 2 * s_b);
    vfloat4 vb4;

    aData += 3 * s_a;
    bData += s_b;

    for (size_t n = 0; n < N; n += 4)
    {
        res1 = vfmaq_laneq_f32(res1, vb1, va1, 0);
        res2 = vfmaq_laneq_f32(res2, vb1, va1, 1);
        res3 = vfmaq_laneq_f32(res3, vb1, va1, 2);
        res4 = vfmaq_laneq_f32(res4, vb1, va1, 3);
        va4 = vld1q_f32(aData);
        vb4 = vld1q_f32(bData);

        res5 = vfmaq_laneq_f32(res5, vb2, va2, 0);
        res6 = vfmaq_laneq_f32(res6, vb2, va2, 1);
        res7 = vfmaq_laneq_f32(res7, vb2, va2, 2);
        res8 = vfmaq_laneq_f32(res8, vb2, va2, 3);
        va1 = vld1q_f32(aData + s_a);
        vb1 = vld1q_f32(bData + s_a);

        res9 = vfmaq_laneq_f32(res9, vb3, va3, 0);
        res10 = vfmaq_laneq_f32(res10, vb3, va3, 1);
        res11 = vfmaq_laneq_f32(res11, vb3, va3, 2);
        res12 = vfmaq_laneq_f32(res12, vb3, va3, 3);
        va2 = vld1q_f32(aData + 2 * s_a);
        vb2 = vld1q_f32(bData + 2 * s_a);

        res13 = vfmaq_laneq_f32(res13, vb4, va4, 0);
        res14 = vfmaq_laneq_f32(res14, vb4, va4, 1);
        res15 = vfmaq_laneq_f32(res15, vb4, va4, 2);
        res16 = vfmaq_laneq_f32(res16, vb4, va4, 3);
        va3 = vld1q_f32(aData + 3 * s_a);
        vb3 = vld1q_f32(bData + 3 * s_a);

        aData += 4 * s_a;
        bData += 4 * s_b;
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

    vst1q_f32(outData, res1);
    vst1q_f32(outData + 1 * s_out, res2);
    vst1q_f32(outData + 2 * s_out, res2);
    vst1q_f32(outData + 3 * s_out, res2);
}