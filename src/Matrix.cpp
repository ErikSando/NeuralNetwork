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

Matrix Matrix::operator*(const Matrix& other) {
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
    // Matrix* MatrixMultiply(Matrix* matrix1, Matrix* matrix2) {
    //     int rows_1 = matrix1->rows;
    //     int cols_1 = matrix1->columns;
    //     int rows_2 = matrix2->rows;
    //     int cols_2 = matrix2->columns;

    //     assert(cols_1 == rows_2);

    //     Matrix* resultant = new Matrix;
    //     resultant->rows = rows_1;
    //     resultant->columns = cols_2;
    //     resultant->elements.resize(rows_1 * cols_2, 0.0f);

    //     for (int row = 0; row < rows_1; row++) {
    //         for (int col = 0; col < cols_2; col++) {
    //             for (int i = 0; i < cols_1; i++) {
    //                 int index1 = row * cols_1 + i;
    //                 int index2 = col + i * cols_2;
    //                 int index_r = row * cols_2 + col;

    //                 assert(index_r < rows_1 * cols_2);

    //                 resultant->elements[index_r] += matrix1->elements[index1] * matrix2->elements[index2]; // using direct indexing for better performance
    //             }
    //         }
    //     }

    //     return resultant;
    // }

    Matrix* MatrixAdd(Matrix* matrix1, Matrix* matrix2) {
        int rows = matrix1->rows;
        int cols = matrix1->columns;

        assert(rows == matrix2->rows && cols == matrix2->columns);

        int size = rows * cols;

        Matrix* resultant = new Matrix;
        resultant->elements.reserve(size);
        resultant->rows = rows;
        resultant->columns = cols;

        for (int i = 0; i < size; i++) {
            resultant->elements.emplace_back(matrix1->elements[i] + matrix2->elements[i]);
        }

        return resultant;
    }

    std::vector<float> RandomMatrixElements(int size, float min, float max) {
        std::vector<float> result;

        for (int _ = 0; _ < size; _++) {
            result.emplace_back(Random::Float(min, max));
        }

        return result;
    }

    void PrintMatrix(Matrix* matrix) {
        for (int c = 0; c < matrix->columns; c++) {
            for (int r = 0; r < matrix->rows; r++) {
                std::cout << " " << matrix->elements[r * matrix->columns + c];
            }

            std::cout << std::endl;
        }
    }

    void PrintMatrix(Matrix& matrix) {
        PrintMatrix(&matrix);
    }
}

// I was writing a more general purpose matrix class, but I want performance so I am making it simpler

// #include <Matrix.h>

// Matrix::Matrix(int rows, int columns) {
//     assert(rows > 0 && columns > 0);

//     elements.resize(rows * columns);
// }

// float Matrix::GetElement(int row, int column) {
//     assert(row <= m_rows && row > 0);
//     assert(column <= m_columns && column > 0);

//     int index = (row - 1) * m_columns + (column - 1);

//     return elements.at(index);
// }

// bool Matrix::SetElement(int row, int column, float value) {
//     assert(row <= m_rows && row > 0);
//     assert(column <= m_columns && column > 0);

//     int index = (row - 1) * m_columns + (column - 1);

//     elements.at(index) = value;

//     return true;
// }

// bool Matrix::SetElements(std::vector<float> values) {
//     assert(values.size() == elements.size());
// }

// int Matrix::GetRows() {
//     return m_rows;
// }

// int Matrix::GetColumns() {
//     return m_columns;
// }