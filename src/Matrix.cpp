#include <iostream>
#include <cassert>

#include <Utility.h>

Matrix Matrix::operator*(const Matrix& other) {
    int rows_2 = other.rows;
    int cols_2 = other.columns;

    assert(columns == rows_2);

    Matrix resultant;
    resultant.rows = rows;
    resultant.columns = cols_2;
    resultant.elements.resize(rows * cols_2, 0.0f);

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols_2; col++) {
            for (int i = 0; i < columns; i++) {
                int index1 = row * columns + i;
                int index2 = col + i * cols_2;
                int index_r = row * cols_2 + col;

                assert(index_r < rows * cols_2);

                resultant.elements[index_r] += elements[index1] * other.elements[index2]; // using direct indexing for better performance
            }
        }
    }

    return resultant;
}

Matrix Matrix::operator+(const Matrix& other) {
    assert(rows == other.rows && columns == other.columns);

    int size = rows * columns;

    Matrix resultant{ .rows = rows, .columns = columns };
    resultant.elements.reserve(size);

    for (int i = 0; i < size; i++) {
        resultant.elements.emplace_back(elements[i] + other.elements[i]);
    }

    return resultant;
}

namespace Utility {
    std::vector<float> RandomMatrixElements(int size, float min, float max) {
        std::vector<float> result;

        for (int _ = 0; _ < size; _++) {
            result.emplace_back(Random::Float(min, max));
        }

        return result;
    }

    void PrintMatrix(Matrix& matrix) {
        for (int c = 0; c < matrix.columns; c++) {
            for (int r = 0; r < matrix.rows; r++) {
                std::cout << " " << matrix.elements[r * matrix.columns + c];
            }

            std::cout << std::endl;
        }
    }
}