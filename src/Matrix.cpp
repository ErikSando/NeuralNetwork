#include <iostream>
#include <cassert>

#include <Utility.h>

void Matrix::Resize() {
    elements.resize(rows * columns, 0.0f);
}

void Matrix::Resize(int r, int c) {
    rows = r;
    columns = c;
    elements.resize(rows * columns, 0.0f);
}

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

std::vector<float> Matrix::RandomMatrixElements(int size, float min, float max) {
    std::vector<float> result;

    for (int _ = 0; _ < size; _++) {
        result.emplace_back(Random::Float(min, max));
    }

    return result;
}

void Matrix::MatrixMultiply(const Matrix& m1, const Matrix& m2, Matrix& output) {
    int rows_1 = m1.rows;
    int cols_1 = m1.columns;
    int rows_2 = m2.rows;
    int cols_2 = m2.columns;

    assert(cols_1 == rows_2 && output.rows == m1.rows && output.columns == m2.columns);

    for (int row = 0; row < rows_1; row++) {
        for (int col = 0; col < cols_2; col++) {
            for (int i = 0; i < cols_1; i++) {
                int index1 = row * cols_1 + i;
                int index2 = col + i * cols_2;
                int index_r = row * cols_2 + col;

                assert(index_r < rows_1 * cols_2);

                output.elements[index_r] += m1.elements[index1] * m2.elements[index2];
            }
        }
    }
}

void Matrix::MatrixAdd(const Matrix& m1, const Matrix& m2, Matrix& output) {
    assert(m1.rows == m2.rows && m1.columns == m2.columns && output.rows == m1.rows && output.columns == m1.columns);

    int size = output.rows * output.columns;

    for (int i = 0; i < size; i++) {
        output.elements[i] = m1.elements[i] + m2.elements[i];
    }
}

void Matrix::MatrixMultiplyResize(const Matrix& m1, const Matrix& m2, Matrix& output) {
    int rows_1 = m1.rows;
    int cols_1 = m1.columns;
    int rows_2 = m2.rows;
    int cols_2 = m2.columns;

    assert(cols_1 == rows_2);

    output.rows = rows_1;
    output.columns = cols_2;
    output.elements.resize(rows_1 * cols_2, 0.0f);

    for (int row = 0; row < rows_1; row++) {
        for (int col = 0; col < cols_2; col++) {
            for (int i = 0; i < cols_1; i++) {
                int index1 = row * cols_1 + i;
                int index2 = col + i * cols_2;
                int index_r = row * cols_2 + col;

                assert(index_r < rows_1 * cols_2);

                output.elements[index_r] += m1.elements[index1] * m2.elements[index2];
            }
        }
    }
}

void Matrix::MatrixAddResize(const Matrix& m1, const Matrix& m2, Matrix& output) {
    assert(m1.rows == m2.rows && m1.columns == m2.columns);

    int size = m1.rows * m1.columns;
    output.rows = m1.rows;
    output.columns = m1.columns;
    output.elements.resize(size, 0.0f);

    for (int i = 0; i < size; i++) {
        output.elements[i] = m1.elements[i] + m2.elements[i];
    }
}

void Matrix::PrintMatrix(Matrix& matrix) {
    for (int r = 0; r < matrix.rows; r++) {
        std::cout << "Row " << r << " -> ";
        
        for (int c = 0; c < matrix.columns; c++) {
            std::cout << " " << matrix.elements[r * matrix.columns + c];
        }

        std::cout << std::endl;
    }
}