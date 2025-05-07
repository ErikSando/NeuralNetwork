#pragma once

#include <iostream>
#include <array>
#include <vector>

#include <Globals.h>

struct Node {
    float value = 0;
    float bias = 0;
};

struct Connection {
    float weight;
    Node* node0;
    Node* node1;
};

struct Layer {
    std::vector<Connection*> connections;
    std::vector<Node*> nodes;
};

struct TestingData {
    int correct = 0;
    int incorrect = 0;
};

class NeuralNetwork {
    public:

    NeuralNetwork();
    ~NeuralNetwork();

    std::array<float, N_OUTPUT_NODES> GetOutputs(const std::array<float, N_INPUT_NODES> inputs);
    
    //static std::array<float, N_OUTPUT_NODES> GetConfidences(const std::array<float, N_OUTPUT_NODES> outputs);

    void Train(const int iterations, const std::string& trainpath, const bool newpath = false);
    // void SetLearningRate(float lr) : learning_rate(lr) {}
    void SetLearningRate(float lr);

    TestingData* Test(const int iterations, const std::string& trainpath, const bool newpath = false);

    bool SaveModel(const std::string& savepath);
    void UploadModel(const std::string& savepath);

    private:

    std::array<Node*, N_INPUT_NODES> input_nodes;
    std::array<Node*, N_HIDDEN_NODES> hidden_nodes;
    std::array<Node*, N_OUTPUT_NODES> output_nodes;
    std::array<Layer*, N_LAYERS> layers; // doesn't include the input layer

    int current_layer = 0;
    float learning_rate = 0.001;
    int current_row = 1; // current row in the training data
    int c_training_row = 1; // current row in the testing data
    int reg_str = 0.0001; // regularisation strength, denoted lambda i believe

    void ClearNodes();
    void AddLayer(Layer* layer);
};