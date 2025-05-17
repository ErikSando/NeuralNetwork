#include <DataParser.h>
#include <NeuralNetwork.h>

// PROBLEM:
// Testing time increases with each test (for the same number of iterations)

TestingData* NeuralNetwork::Test(const int iterations, const std::string& testpath, const bool newpath) {
    if (newpath) c_testing_row = 1;

    TestingData* t_data = new TestingData;

    for (int i = 0; i < iterations; i++) {
        std::vector<ImageData*> image_data = DataParser::GetRowsImageData(c_testing_row, 1, testpath);
        ImageData* data = image_data[0];

        std::array<float, N_OUTPUT_NODES> outputs = GetOutputs(data->pixels);

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

        for (int i = 0; i < image_data.size(); i++) {
            delete image_data[i];
        }

        image_data.clear();
    }

    return t_data;
}