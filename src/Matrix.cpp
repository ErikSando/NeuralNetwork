#include <iostream>
#include <cassert>

#include <Utility.h>

namespace Utility {
    Matrix* MatrixMultiply(Matrix* matrix1, Matrix* matrix2) {
        int rows_1 = matrix1->rows;
        int cols_1 = matrix1->columns;
        int rows_2 = matrix2->rows;
        int cols_2 = matrix2->columns;

        assert(rows_1 == cols_2);

        Matrix* resultant = new Matrix;
        resultant->rows = rows_1;
        resultant->columns = cols_2;
        resultant->elements.resize(rows_1 * cols_2);

        // this iterates through each row, which is the same as the number of columns in matrix2
        for (int row = 0; row < rows_1; row++) {
            for (int col = 0; col < cols_2; col++) {
                for (int i = 0; i < cols_1; i++) {
                    int index1 = row * cols_1 + i;
                    int index2 = col + i * cols_2;
                    int index_r = row * cols_2 + col;

                    assert(index_r < rows_1 * cols_2);

                    // not using .at() to save speed
                    resultant->elements[index_r] += matrix1->elements[index1] * matrix2->elements[index2];
                }
            }
        }

        return resultant;
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