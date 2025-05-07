#include <iostream>

#include <Globals.h>
#include <Utility.h>

int main() {
    Random::Init();

    StartConsoleLoop();

    // Matrix m1{ .rows = 2, .columns = 3 };
    // Matrix m2{ .rows = 3, .columns = 2 };

    // m1.elements = { 2, 1, 3, 2, 4, 1 };
    // m2.elements = { 3, 4, 1, 2, 2, 3 };

    // Matrix* mr = Utility::MatrixMultiply(&m1, &m2);

    // for (int c = 0; c < mr->columns; c++) {
    //     for (int r = 0; r < mr->rows; r++) {
    //         int index = r * mr->rows + c;
    //         std::cout << " " << mr->elements.at(index);
    //     }

    //     std::cout << std::endl;
    // }

    return 0;
}