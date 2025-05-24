#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

#include <Utility.h>

const float epsilon = 1e-15f;

namespace Random {
    void Init() {
        std::srand(time(nullptr));
    }
    
    float Float() {
        return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    float Float(float min, float max) {
        return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * (max - min) + min;
    }
}

namespace Utility {
    // using 32 bit int instead of streampos to save memory, maximum index cannot exceed 32 bits
    // stored in static memory so the large size doesn't matter
    static std::array<uint32_t, TRAINING_ROWS> line_offsets;

    static std::string lastpath;
    //static std::streampos jump = 0;
    static int last_line = 0;
    static int largest_line = 1;

    // this seems faster than the other one (tested with train 400, this had 14288ms, the other 16261ms)
    std::string ReadLine(const int line, const std::string& filepath) {
        std::ifstream input(filepath);

        if (!input.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return "";
        }

        if (filepath != lastpath) {
            last_line = 0;
            largest_line = 1;
            lastpath = filepath;
        }

        int current_line = 1;

        if (line <= largest_line) {
            assert(line >= 1 && line <= TRAINING_ROWS);

            current_line = line;
            std::streampos jump = static_cast<std::streampos>(line_offsets[line - 1]);
            input.seekg(jump);
        }
        else {
            assert(largest_line >= 1 && largest_line <= TRAINING_ROWS);

            current_line = largest_line;
            std::streampos jump = static_cast<std::streampos>(line_offsets[current_line - 1]);
            input.seekg(jump);
        }

        std::string line_string;

        for (; current_line <= line; current_line++) {
            if (!std::getline(input, line_string)) {
                throw std::out_of_range("Requested line number out of range");
            }

            std::streampos pos = input.tellg();

            if (pos != EOF) {
                //jump = pos;
                line_offsets[current_line] = static_cast<uint32_t>(pos);

                if (current_line > largest_line) {
                    largest_line = current_line;
                }
            }
        }

        // for (int i = 0; i < line; i++) {
        //     std::cout << "Offset for line " << i << ": " << line_offsets[i] << std::endl;
        // }

        last_line = line;

        return line_string;
    }

    // static std::streampos jump = 0;

    // std::string ReadLine(const int line, const std::string& filepath) {
    //     std::ifstream input(filepath);

    //     if (!input.is_open()) {
    //         std::cerr << "Failed to open file: " << filepath << std::endl;
    //         return "";
    //     }

    //     if (filepath != lastpath) {
    //         jump = 0;
    //         last_line = 0;
    //         largest_line = 1;
    //         lastpath = filepath;
    //     }

    //     if (last_line < line) {
    //         input.seekg(jump);
    //     }

    //     std::string line_string;

    //     for (int current_line = last_line + 1; current_line <= line; current_line++) {
    //         if (!std::getline(input, line_string)) {
    //             throw std::out_of_range("Requested line number out of range");
    //         }

    //         std::streampos pos = input.tellg();

    //         if (pos != EOF) {
    //             jump = pos;
    //         }
    //     }

    //     last_line = line;

    //     return line_string;
    // }

    std::string ReadFile(const std::string& filepath) {
        std::ifstream input(filepath);
    
        if (!input.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return "";
        }

        std::stringstream contents;

        contents << input.rdbuf();
        input.close();

        return contents.str();
    }

    void GetTrueOutputs(int digit, std::array<uint8_t, N_OUTPUT_NODES>& true_outputs) {
        assert(digit >= 0 && digit < N_OUTPUT_NODES);
        true_outputs[digit] = 1;
    }

    void GetBatchedTrueOutputs(std::array<int, BATCH_SIZE> digits, std::array<uint8_t, N_OUTPUT_NODES * BATCH_SIZE>& true_outputs) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            assert(digits[i] >= 0 && digits[i] < N_OUTPUT_NODES);
            true_outputs[digits[i] + i * N_OUTPUT_NODES] = 1;
        }
    }

    namespace Activation {
        float Sigmoid(float z) {
            return 1.0f / (1.0f + std::exp(-z));
        }
    
        float ReLU(float z) {
            return std::fmax(0.0f, z);
        }

        float LeakyReLU(float z) {
            if (z > 0) return z;
            else return 0.01 * z;
        }

        float Tanh(float z) {
            return (std::exp(z) - std::exp(-z)) / (std::exp(z) + std::exp(-z));
        }
    }

    namespace Loss {
        float CategoricalCrossEntropy(std::array<uint8_t, N_OUTPUT_NODES> true_outputs, std::array<float, N_OUTPUT_NODES> softmax_outputs) {
            float loss = 0.0f;
            
            for (int i = 0; i < N_OUTPUT_NODES; i++) {
                float y = softmax_outputs[i];
                if (y == 0.0f) y = epsilon;
                loss += static_cast<float>(true_outputs[i]) * -std::log(y);
            }

            return loss;
        }

        float MeanSquaredError(std::array<uint8_t, N_OUTPUT_NODES> true_outputs, std::array<float, N_OUTPUT_NODES> softmax_outputs) {
            float mse = 0.0f;

            for (int i = 0; i < N_OUTPUT_NODES; i++) {
                float error = true_outputs[i] - softmax_outputs[i];
                mse += error;
            }

            return mse / N_OUTPUT_NODES;
        }
    }
}