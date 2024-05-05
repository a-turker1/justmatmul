#define STORE_RESULTS(index, res1, res2) \
    vst1q_f32(outData + (index) * s_out, res1); \
    vst1q_f32(outData + (index) * s_out + 4, res2);


#define vfloat4 float32x4_t


/*
Applies operation A@B=OUT.
Expected A (mxn) transposed, B(NXK) row major and out (MXK) row major.
*/

void matmul_12x8_micro_kernel(int N, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out)
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
    vfloat4 vb1_1 = vld1q_f32(bData);
    vfloat4 va2 = vld1q_f32(aData + 4);
    vfloat4 vb2_1 = vld1q_f32(bData + 4);
    vfloat4 va3 = vld1q_f32(aData + 8);

    aData += s_a;
    bData += s_b;

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



void matmul_4x4_micro_kernel(int N, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out){

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
    vfloat4 va2 = vld1q_f32(aData+s_a);
    vfloat4 va3 = vld1q_f32(aData+2*s_a);
    vfloat4 va4;

    vfloat4 vb1 = vld1q_f32(bData);
    vfloat4 vb2 = vld1q_f32(bData+s_b);
    vfloat4 vb3 = vld1q_f32(bData+2*s_b);
    vfloat4 vb4;

    aData += 3*s_a;
    bData += s_b;


    for (size_t n = 0; n < N; n+=4)
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
        va1 = vld1q_f32(aData+s_a);
        vb1 = vld1q_f32(bData+s_a);

        res9 = vfmaq_laneq_f32(res9, vb3, va3, 0);
        res10 = vfmaq_laneq_f32(res10, vb3, va3, 1);
        res11 = vfmaq_laneq_f32(res11, vb3, va3, 2);
        res12 = vfmaq_laneq_f32(res12, vb3, va3, 3);
        va2 = vld1q_f32(aData+2*s_a);
        vb2 = vld1q_f32(bData+2*s_a);

        res13 = vfmaq_laneq_f32(res13, vb4, va4, 0);
        res14 = vfmaq_laneq_f32(res14, vb4, va4, 1);
        res15 = vfmaq_laneq_f32(res15, vb4, va4, 2);
        res16 = vfmaq_laneq_f32(res16, vb4, va4, 3);
        va3 = vld1q_f32(aData+3*s_a);
        vb3 = vld1q_f32(bData+3*s_a);

        aData += 4*s_a;
        bData += 4*s_b;
    }

    res1 = vaddq_f32(res1, res5);
    res9 = vaddq_f32(res9, res13);
    res2 = vaddq_f32(res2, res6);
    res10 = vaddq_f32(res10, res14);
    res3 = vaddq_f32(res3, res7);
    res11 = vaddq_f32(res11, res15);
    res4 = vaddq_f32(res4, res8);
    res12 = vaddq_f32(res12, res16);
    res1 = vaddq_f32(res1,res9);
    res2 = vaddq_f32(res2,res10);
    res3 = vaddq_f32(res3,res11);
    res4 = vaddq_f32(res4,res12);

    vst1q_f32(outData, res1);
    vst1q_f32(outData + 1 * s_out, res2);
    vst1q_f32(outData + 2 * s_out, res2);
    vst1q_f32(outData + 3 * s_out, res2);
}