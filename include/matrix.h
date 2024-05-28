#pragma once
#include <iostream>
struct Matrix
{
    int rows;
    int cols;
    bool is_row_major;
    float *base;

    // Constructor to initialize the matrix with given dimensions
    Matrix(int rows_, int cols_, bool row_major = false) : rows(rows_), cols(cols_), is_row_major(row_major)
    {
        // Allocate memory for the matrix data
        base = new float[rows_ * cols_];
        for (size_t i = 0; i < rows_ * cols_; i++)
        {

            base[i] = row_major ? i : (i % rows_) * cols_ + i / rows_;
        }
    }

    Matrix(int rows_, int cols_, float val, bool row_major = false) : rows(rows_), cols(cols_), is_row_major(row_major)
    {
        base = new float[rows_ * cols_];
        for (size_t i = 0; i < rows_ * cols_; i++)
        {
            base[i] = val;
        }
    }

    Matrix(Matrix&& other) noexcept
        : rows(other.rows), cols(other.cols), is_row_major(other.is_row_major), base(other.base) {
        other.base = nullptr;
    }

    // Move assignment operator
    Matrix& operator=(Matrix&& other) noexcept {
        if(base != other.base){
            delete[] base;
        }


        rows = other.rows;
        cols = other.cols;
        is_row_major = other.is_row_major;
        base = other.base;
        other.base = nullptr;

        return *this;
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

    Matrix transpose()
    {
    Matrix transposed = Matrix(cols, rows, is_row_major);
    auto transposed_data = transposed.data();
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (is_row_major)
            {
                transposed_data[j * rows + i] = base[i * cols + j];
            }
            else
            {
                transposed_data[i * cols + j] = base[j * rows + i];
            }
        }
    }
    return transposed;
    }

    friend std::ostream &operator<<(std::ostream &os, const Matrix &mat)
    {
        if (mat.is_row_major)
        {
            os << "[";
            for (int i = 0; i < mat.rows; ++i)
            {

                if (mat.rows > 6 && i == 3)
                {
                    os << "..., " << std::endl;
                    i = mat.rows - 4;
                    continue;
                }

                os << "[";
                for (int j = 0; j < mat.cols; ++j)
                {
                    if (mat.cols > 6 && j == 3)
                    {
                        os << "..., ";
                        j = mat.cols - 4;
                        continue;
                    }

                    os << mat.base[j + mat.cols * i];
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
        else{
            os << "[";
            for (int i = 0; i < mat.rows; ++i)
            {

                if (mat.rows > 6 && i == 3)
                {
                    os << "..., " << std::endl;
                    i = mat.rows - 4;
                    continue;
                }

                os << "[";
                for (int j = 0; j < mat.cols; ++j)
                {
                    if (mat.cols > 6 && j == 3)
                    {
                        os << "..., ";
                        j = mat.cols - 4;
                        continue;
                    }

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
    }
};