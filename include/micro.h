#define STORE_RESULTS(index, res1, res2) \
    vst1q_f32(outData + (index) * s_out, res1); \
    vst1q_f32(outData + (index) * s_out + 4, res2);


void matmul_12x8_micro_kernel(int N, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out)
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
    float32x4_t va3 = vld1q_f32(aData + 8);

    aData += s_a;
    bData += s_b;

    #pragma unroll 4
    for (size_t n = 0; n < N; n++) // n
    {
        res1 = vfmaq_laneq_f32(res1, vb1_1, va1, 0);
        res2 = vfmaq_laneq_f32(res2, vb2_1, va1, 0); // Row1

         res3 = vfmaq_laneq_f32(res3, vb1_1, va1, 1);
        res4 = vfmaq_laneq_f32(res4, vb2_1, va1, 1); // Row2

        res5 = vfmaq_laneq_f32(res5, vb1_1, va1, 2);
        res6 = vfmaq_laneq_f32(res6, vb2_1, va1, 2); // Row3

        res7 = vfmaq_laneq_f32(res7, vb1_1, va1, 3);
        res8 = vfmaq_laneq_f32(res8, vb2_1, va1, 3); // Row4
        va1 = vld1q_f32(aData);

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
        vb1_1 = vld1q_f32(bData);
        vb2_1 = vld1q_f32(bData + 4);
        va3 = vld1q_f32(aData + 8);
        aData += s_a;
        bData += s_b;
    }

    STORE_RESULTS(0, res1, res2)
    STORE_RESULTS(1, res3, res4)
    STORE_RESULTS(2, res5, res6)
    STORE_RESULTS(3, res7, res8)
    STORE_RESULTS(4, res9, res10)
    STORE_RESULTS(5, res11, res12)
    STORE_RESULTS(6, res13, res14)
    STORE_RESULTS(7, res15, res16)
    STORE_RESULTS(8, res17, res18)
    STORE_RESULTS(9, res19, res20)
    STORE_RESULTS(10, res21, res22)
    STORE_RESULTS(11, res23, res24)
}

