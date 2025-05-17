#pragma once

#include <vector>

struct Matrix {
    std::vector<float> elements{};
    int rows;
    int columns;

    Matrix operator*(const Matrix& other);
    Matrix operator+(const Matrix& other);
};