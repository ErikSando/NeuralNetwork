#include <DataParser.h>
#include <NeuralNetwork.h>

// PROBLEM:
// Testing time increases with each test (for the same number of iterations)

TestingData* NeuralNetwork::Test(const int iterations, const std::string& testpath, const bool newpath) {
    if (newpath) c_testing_row = 1;

    TestingData* t_data = new TestingData;

    for (int i = 0; i < iterations; i++) {
        std::array<uint8_t, N_INPUT_NODES * BATCH_SIZE> inputs;
        const ImageData* data = DataParser::GetRowData(c_testing_row, testpath);

        for (int i = 0; i < BATCH_SIZE; i++) {
            std::copy(data->pixels.begin(), data->pixels.end(), inputs.begin() + i * BATCH_SIZE * sizeof(uint8_t));
        }

        std::array<float, N_OUTPUT_NODES * BATCH_SIZE> outputs = GetOutputs(inputs);

        c_testing_row = (c_testing_row % TESTING_ROWS) + 1;
        assert(c_testing_row <= TESTING_ROWS);

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