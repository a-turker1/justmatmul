#pragma once
#include "matrix.h"
void naive_matmul(Matrix& a, Matrix& b, Matrix& out);
void neon_matmul(Matrix &a, Matrix &b, Matrix &out);
void matmul_8x12_micro_kernel(int N, float* aData, float* bData, float*outData, int s_a, int s_b, int s_out);