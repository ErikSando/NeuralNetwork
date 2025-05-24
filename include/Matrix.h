#pragma once

#include <vector>

struct Matrix {
    void Resize();
    void Resize(int r, int c);

    std::vector<float> elements{};
    int rows;
    int columns;

    Matrix operator*(const Matrix& other);
    Matrix operator+(const Matrix& other);

    static std::vector<float> RandomMatrixElements(int size, float min, float max);
    static void MatrixMultiply(const Matrix& m1, const Matrix& m2, Matrix& output);
    static void MatrixAdd(const Matrix& m1, const Matrix& m2, Matrix& output);
    static void MatrixMultiplyResize(const Matrix& m1, const Matrix& m2, Matrix& output);
    static void MatrixAddResize(const Matrix& m1, const Matrix& m2, Matrix& output);
    static void PrintMatrix(Matrix& matrix);
};