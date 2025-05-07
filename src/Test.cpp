#include <DataParser.h>
#include <NeuralNetwork.h>

// PROBLEM:
// Testing time increases with each test (for the same number of iterations)

TestingData* NeuralNetwork::Test(const int iterations, const std::string& testpath, const bool newpath) {
    if (newpath) c_training_row = 1;

    TestingData* t_data = new TestingData;

    int correct = 0;
    int incorret = 0;

    for (int i = 0; i < iterations; i++) {
        ImageData* data = DataParser::GetRowImageData(c_training_row, testpath);

        

        delete data;
    }

    return t_data;
}