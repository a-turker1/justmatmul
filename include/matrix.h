#pragma once
#include <iostream>
struct Matrix
{
    int rows;
    int cols;
    float *base;

    // Constructor to initialize the matrix with given dimensions
    Matrix(int h, int w) : cols(h), rows(w)
    {
        // Allocate memory for the matrix data
        base = new float[h * w];
        for (size_t i = 0; i < h*w; i++)
        {
            base[i] = 1.0f;
        }
        
    }

    ~Matrix()
    {
        delete[] base;
    }

    float *operator[](int idx) const
    {
        return &base[idx];
    }

    float *data()
    {
        return base;
    }

    friend std::ostream &operator<<(std::ostream &os, const Matrix &mat)
    {
        os << "[";
        for (int i = 0; i < mat.rows; ++i)
        {
            os << "[";
            for (int j = 0; j < mat.cols; ++j)
            {
                os << mat.base[i * mat.cols + j];
                if (j < mat.cols - 1)
                {
                    os << ", ";
                }
            }
            os << "]";
            if (i < mat.rows - 1)
            {
                os << std::endl;
            }
        }
        os << "]";
        return os;
    }
};