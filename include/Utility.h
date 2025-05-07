#pragma once

#include <array>
#include <string>
#include <vector>

#include <Globals.h>
#include <Matrix.h>

namespace Random {
    void Init();
    float Float();
    float Float(float min, float max);
}

namespace Utility {
    std::string ReadLine(const int line, const std::string& filepath);
    std::string ReadFile(const std::string& filepath);
    std::array<float, N_OUTPUT_NODES> GetTrueOutputs(int digit);

    std::vector<float> RandomMatrixElements(int size, float min, float max);
    void PrintMatrix(Matrix& matrix);

    namespace Activation {
        float Sigmoid(float z);
        float ReLU(float z);
        float Tanh(float z);
    }
    
    namespace Loss {
        float CategoricalCrossEntropy(std::array<float, N_OUTPUT_NODES> true_outputs, std::array<float, N_OUTPUT_NODES> softmax_outputs);
        float MeanSquaredError(std::array<float, N_OUTPUT_NODES> true_outputs, std::array<float, N_OUTPUT_NODES> softmax_outputs);
    }
}