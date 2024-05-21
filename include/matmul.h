#pragma once
#include "matrix.h"
void naive_matmul(Matrix& a, Matrix& b, Matrix& out);
void neon_matmul(Matrix &a, Matrix &b, Matrix &out);
void naive_matmul_4x1(Matrix &a, Matrix &b, Matrix &out);
void naive_matmul_4x4(Matrix &a, Matrix &b, Matrix &out);
void matmul_4x4_neon(Matrix &a, Matrix &b, Matrix &out);
void matmul_4x4_neon_2(Matrix &a, Matrix &b, Matrix &out);
void matmul_4x4_neon_3(Matrix &a, Matrix &b_, Matrix &out);
void matmul_4x4_neon_4(Matrix &a, Matrix &b_, Matrix &out);
void matmul_4x4_neon_5(Matrix &a, Matrix &b_, Matrix &out);
void matmul_12x8_neon(Matrix &a, Matrix &b_, Matrix &out);