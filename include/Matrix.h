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
};