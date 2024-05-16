#pragma once
#include <iostream>
struct Matrix
{
    int rows;
    int cols;
    float *base;

    // Constructor to initialize the matrix with given dimensions
    Matrix(int rows_, int cols_) : rows(rows_), cols(cols_)
    {
        // Allocate memory for the matrix data
        base = new float[rows_ * cols_];
        for (size_t i = 0; i < rows_*cols_; i++)
        {
            base[i] = 1;
        }
        
    }

    Matrix(int rows_, int cols_, float val) : rows(rows_), cols(cols_)
    {
        base = new float[rows_ * cols_];
        for (size_t i = 0; i < rows_*cols_; i++)
        {
            base[i] = val;
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

    Matrix transpose(){
        Matrix transposed = Matrix(cols, rows);
        auto data = transposed.data();
        for (size_t i = 0; i < cols*rows; i++)
        {
            int _row = i/cols;
            int _col = i%cols;
            data[rows*_col + _row] = base[i];
        }
        return transposed;

    }

    friend std::ostream &operator<<(std::ostream &os, const Matrix &mat)
    {
        os << "[";
        for (int i = 0; i < mat.rows; ++i)
        {
            os << "[";
            for (int j = 0; j < mat.cols; ++j)
            {
                os << mat.base[i + mat.rows * j];
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