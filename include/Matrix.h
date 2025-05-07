#pragma once

#include <vector>

struct Matrix {
    std::vector<float> elements{};
    int rows;
    int columns;

    Matrix operator*(const Matrix& other);
    Matrix operator+(const Matrix& other);
};

// I was writing a more general purpose matrix class, but I want performance so I am making it simpler

// class Matrix {
//     public:

//     std::vector<float> elements;

//     Matrix(int rows, int columns) : m_rows(rows), m_columns(columns) {};

//     float GetElement(int row, int column);

//     bool Matrix::SetElement(int row, int column, float value);
//     bool Matrix::SetElements(std::vector<float> values);

//     int GetRows();
//     int GetColumns();

//     private:

//     int m_rows;
//     int m_columns;
// };

// extern Matrix& MultiplyMatrices(Matrix& matrix1, Matrix& matrix2);