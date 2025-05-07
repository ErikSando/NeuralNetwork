#pragma once

#include <iostream>
#include <array>
#include <vector>

#include <Globals.h>
#include <Matrix.h>

struct TestingData {
    int correct = 0;
    int incorrect = 0;
};

class NeuralNetwork {
    public:

    NeuralNetwork();
    ~NeuralNetwork();

    //std::array<float, N_INPUT_NODES> NormaliseInputs(const std::array<float, N_INPUT_NODES> inputs);
    std::array<float, N_OUTPUT_NODES> GetOutputs(const std::array<float, N_INPUT_NODES> inputs);
    
    //static std::array<float, N_OUTPUT_NODES> GetConfidences(const std::array<float, N_OUTPUT_NODES> outputs);

    void Train(const int iterations, const std::string& trainpath, const bool newpath = false);
    void SetLearningRate(const float lr);
    void SetBatchSize(const int bs);

    TestingData* Test(const int iterations, const std::string& trainpath, const bool newpath = false);

    bool SaveModel(const std::string& savepath);
    void UploadModel(const std::string& savepath);

    private:

    Matrix h1_weights;
    Matrix h2_weights;
    Matrix out_weights;

    Matrix h1_biases;
    Matrix h2_biases;
    Matrix out_biases;

    Matrix h1_nodes;
    Matrix h2_nodes;
    Matrix out_nodes;

    float learning_rate = 0.01;
    int batch_size = 32;
    int current_row = 1; // current row in the training data
    int c_training_row = 1; // current row in the testing data
    int reg_str = 0.0001; // regularisation strength, denoted lambda i believe
};