#include <iostream>

#include <Globals.h>
#include <Utility.h>

#include <NeuralNetwork.h>
#include <DataParser.h>

int main() {
    Random::Init();

    StartConsoleLoop();

    // Matrix matrix1;
    // Matrix matrix2;

    // matrix1.elements.resize(6);
    // matrix1.rows = 2;
    // matrix1.columns = 3;

    // matrix2.elements.resize(6);
    // matrix2.rows = 3;
    // matrix2.columns = 2;

    // matrix1.elements = { 2, 1, 3, 2, 4, 1 };
    // matrix2.elements = { 3, 4, 1, 2, 2, 3 };

    // Matrix r = matrix1 * matrix2;

    // std::cout << "Got resultant" << std::endl;

    // Utility::PrintMatrix(r);

    // NeuralNetwork network;

    // ImageData* data = DataParser::ParseInputFile(3, "./data/three");

    // std::array<float, N_OUTPUT_NODES> outputs = network.GetOutputs(data->pixels);

    // // repeated code, TODO: prevent repeating
    // float largest = 0;
    // int l_digit = -1;

    // for (int i = 0; i < N_OUTPUT_NODES; i++) {
    //     std::cout << i << ": " << outputs[i] << std::endl;

    //     if (outputs[i] > largest) {
    //         largest = outputs[i];
    //         l_digit = i;
    //     }
    // }

    // std::cout << "Identified digit: " << l_digit << " (" << largest << " probability)" << std::endl;

    // std::array<float, N_OUTPUT_NODES> true_outputs = Utility::GetTrueOutputs(data->digit);

    // #ifndef NDEBUG

    // for (int i = 0; i < 10; i++) {
    //     float expected = (i == data->digit) ? 1.0f : 0.0f;
    //     assert(true_outputs[i] == expected);
    // }

    // #endif

    // float loss = Utility::Loss::CategoricalCrossEntropy(true_outputs, outputs);
    // std::cout << "CCE Loss: " << loss << " (lower means more accurate)" << std::endl;

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