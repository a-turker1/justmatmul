#pragma once
#include "matrix.h"
void naive_matmul(Matrix& a, Matrix& b, Matrix& out);
void neon_matmul(Matrix &a, Matrix &b, Matrix &out);