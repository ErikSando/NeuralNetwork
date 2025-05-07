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

        std::array<float, N_OUTPUT_NODES> outputs = GetOutputs(data->pixels);

        c_training_row = (c_training_row % TESTING_ROWS) + 1;
        assert(c_training_row <= TESTING_ROWS);

        // repeated code, TODO: prevent repeating
        float largest = 0;
        int l_digit = -1;

        for (int i = 0; i < N_OUTPUT_NODES; i++) {
            if (outputs[i] > largest) {
                largest = outputs[i];
                l_digit = i;
            }
        }

        #ifndef NDEBUG

        //std::cout << "Prediction: " << l_digit << " (" << data->digit << ")" << std::endl;

        #endif

        if (l_digit == data->digit) t_data->correct++;
        else t_data->incorrect++;

        delete data;
    }

    return t_data;
}