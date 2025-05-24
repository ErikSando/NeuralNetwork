#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "Globals.h"
#include "Matrix.h"

namespace Random {
    void Init();
    float Float();
    float Float(float min, float max);
}

namespace Utility {
    std::string ReadLine(const int line, const std::string& filepath);
    std::string ReadFile(const std::string& filepath);
    void GetTrueOutputs(int digit, std::array<uint8_t, N_OUTPUT_NODES>& true_outputs);
    void GetBatchedTrueOutputs(std::array<int, BATCH_SIZE> digits, std::array<uint8_t, N_OUTPUT_NODES * BATCH_SIZE>& true_outputs);

    namespace Activation {
        float Sigmoid(float z);
        float ReLU(float z);
        float LeakyReLU(float z);
        float Tanh(float z);
    }
    
    namespace Loss {
        float CategoricalCrossEntropy(std::array<uint8_t, N_OUTPUT_NODES> true_outputs, std::array<float, N_OUTPUT_NODES> softmax_outputs);
        float MeanSquaredError(std::array<uint8_t, N_OUTPUT_NODES> true_outputs, std::array<float, N_OUTPUT_NODES> softmax_outputs);
    }
}