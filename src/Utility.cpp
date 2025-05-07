#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include <Utility.h>

const float epsilon = 1e-15f;

namespace Random {
    void Init() {
        std::srand(time(nullptr));
    }
    
    float Float() {
        return static_cast<float> (std::rand()) / static_cast<float> (RAND_MAX);
    }

    float Float(float min, float max) {
        return static_cast<float> (std::rand()) / static_cast<float> (RAND_MAX) * (max - min) + min;
    }
}

namespace Utility {
    std::string ReadLine(const int line, const std::string& filepath) {
        std::ifstream input(filepath);
    
        if (input.is_open()) {
            std::string line_string;
    
            int current_line = 1;
    
            while (std::getline(input, line_string)) {
                if (current_line == line) {
                    return line_string;
                }
    
                current_line++;
            }
    
            throw std::out_of_range("Requested line number out of range");
        }
        else {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return "";
        }
    }

    std::string ReadFile(const std::string& filepath) {
        std::ifstream input(filepath);
    
        if (input.is_open()) {
            std::stringstream contents;
    
            contents << input.rdbuf();
            input.close();
    
            return contents.str();
        }
        else {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return "";
        }
    }

    std::array<float, N_OUTPUT_NODES> GetTrueOutputs(int digit) {
        assert(digit >= 0 && digit < N_OUTPUT_NODES);

        std::array<float, N_OUTPUT_NODES> true_outputs{};
        true_outputs[digit] = 1.0f;

        return true_outputs;
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
        float CategoricalCrossEntropy(std::array<float, N_OUTPUT_NODES> true_outputs, std::array<float, N_OUTPUT_NODES> softmax_outputs) {
            float loss = 0.0f;
            
            for (int i = 0; i < N_OUTPUT_NODES; i++) {
                float y = softmax_outputs[i];
                if (y == 0.0f) y = epsilon;
                loss += true_outputs[i] * -std::log(y);
            }

            return loss;
        }

        float MeanSquaredError(std::array<float, N_OUTPUT_NODES> true_outputs, std::array<float, N_OUTPUT_NODES> softmax_outputs) {
            float mse = 0.0f;

            for (int i = 0; i < N_OUTPUT_NODES; i++) {
                float error = true_outputs[i] - softmax_outputs[i];
                mse += error;
            }

            return mse / N_OUTPUT_NODES;
        }
    }
}