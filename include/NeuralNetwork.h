#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

#include "Globals.h"
#include "Matrix.h"

struct TestingData {
    int correct = 0;
    int incorrect = 0;
};

class NeuralNetwork {
    public:

    NeuralNetwork();
    ~NeuralNetwork();

    //std::array<float, N_OUTPUT_NODES> GetOutputs(const std::array<uint8_t, N_INPUT_NODES> inputs, int n_batches = -1);
    std::array<float, N_OUTPUT_NODES * BATCH_SIZE> GetOutputs(const std::array<uint8_t, N_INPUT_NODES * BATCH_SIZE> inputs);

    void Train(const int batches, const std::string& trainpath, const bool newpath = false);
    //void Train(const int iterations, const std::string& trainpath, const bool newpath = false);
    void SetLearningRate(const float lr);
    //void SetBatchSize(const int bs);

    TestingData* Test(const int iterations, const std::string& trainpath, const bool newpath = false);

    bool SaveModel(const std::string& savepath);
    void LoadModel(const std::string& savepath);

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
    //int batch_size = 32;
    int current_row = 1; // current row in the training data
    int c_testing_row = 1; // current row in the testing data
    int reg_str = 0.0001; // regularisation strength, denoted lambda i believe
};