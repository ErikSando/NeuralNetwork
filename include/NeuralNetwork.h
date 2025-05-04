#pragma once

#include <iostream>

#include <vector>
#include <array>

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

class NeuralNetwork {
    public:

    NeuralNetwork();
    ~NeuralNetwork();

    std::array<float, N_OUTPUT_NODES> GetOutputs(const std::array<float, N_INPUT_NODES> inputs);
    
    //static std::array<float, N_OUTPUT_NODES> GetConfidences(const std::array<float, N_OUTPUT_NODES> outputs);

    void Train();

    bool SaveModel(const std::string& savepath);
    void UploadModel(const std::string& savepath);

    private:

    std::array<Node*, N_INPUT_NODES> input_nodes;
    std::array<Node*, N_HIDDEN_NODES> hidden_nodes;
    std::array<Node*, N_OUTPUT_NODES> output_nodes;
    std::array<Layer*, N_LAYERS> layers; // doesn't include the input layer

    int current_layer = 0;

    void ClearNodes();
    void AddLayer(Layer* layer);
};