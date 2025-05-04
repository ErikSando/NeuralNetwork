#pragma once

#include <array>
#include <vector>
#include <string>

#include <Globals.h>

namespace Random {
    void Init();
    float Float();
    float Float(float min, float max);
}

namespace Utility {
    std::string ReadLine(const int line, const std::string& filepath);
    std::string ReadFile(const std::string& filepath);
    std::array<float, N_OUTPUT_NODES> GetTrueOutputs(int digit);

    namespace ActivationFunctions {
        float Sigmoid(float z);
        float ReLU(float z);
        float Tanh(float z);
    }
    
    namespace LossFunctions {
        float CategoricalCrossEntropy(std::array<float, N_OUTPUT_NODES> true_outputs, std::array<float, N_OUTPUT_NODES> softmax_outputs);
        float MeanSquaredError(std::array<float, N_OUTPUT_NODES> true_outputs, std::array<float, N_OUTPUT_NODES> softmax_outputs);
    }
}