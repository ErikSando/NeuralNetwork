#include <cmath>

#include <DataParser.h>
#include <NeuralNetwork.h>
#include <Utility.h>

NeuralNetwork::NeuralNetwork() {
    for (int a = 0; a < N_INPUT_NODES; a++) {
        Node* input_node = new Node;
        input_nodes[a] = input_node;
    }

    std::array<Node*, HIDDEN_LAYER_1_SIZE> l1_hidden_nodes;

    Layer* hidden_layer_1 = new Layer;

    float weight_max = 6 / std::sqrt(N_INPUT_NODES + N_OUTPUT_NODES);
    float weight_min = -weight_max;

    int hidden_index = 0;

    // TODO: find a smarter way to create hidden layers, so I can adjust settings in Globals.h without rewriting the code here and it will work

    for (int a = 0; a < HIDDEN_LAYER_1_SIZE; a++) {
        Node* hidden_node = new Node;

        hidden_nodes[hidden_index++] = hidden_node;
        hidden_layer_1->nodes.emplace_back(hidden_node);
        l1_hidden_nodes[a] = hidden_node;

        // connect with every input node
        for (int b = 0; b < N_INPUT_NODES; b++) {
            Connection* connection = new Connection;
            connection->weight = Random::Float(weight_min, weight_max);
            connection->node0 = input_nodes[b];
            connection->node1 = hidden_node;

            hidden_layer_1->connections.emplace_back(connection);
        }
    }

    AddLayer(hidden_layer_1);

    std::array<Node*, HIDDEN_LAYER_2_SIZE> l2_hidden_nodes;

    Layer* hidden_layer_2 = new Layer;

    for (int b = 0; b < HIDDEN_LAYER_2_SIZE; b++) {
        Node* hidden_node = new Node;

        hidden_nodes[hidden_index++] = hidden_node;
        hidden_layer_2->nodes.emplace_back(hidden_node);
        l2_hidden_nodes[b] = hidden_node;

        // connect with every hidden node in the previous layer
        for (int c = 0; c < HIDDEN_LAYER_1_SIZE; c++) {
            Connection* connection = new Connection;
            connection->weight = Random::Float(weight_min, weight_max);
            connection->node0 = l1_hidden_nodes[c];
            connection->node1 = hidden_node;

            hidden_layer_2->connections.emplace_back(connection);
        }
    }

    AddLayer(hidden_layer_2);

    Layer* output_layer = new Layer;

    for (int a = 0; a < N_OUTPUT_NODES; a++) {
        Node* output_node = new Node;

        output_nodes[a] = output_node;
        output_layer->nodes.emplace_back(output_node);

        for (int b = 0; b < HIDDEN_LAYER_2_SIZE; b++) {
            Connection* connection = new Connection;
            connection->weight = Random::Float(weight_min, weight_max);
            connection->node0 = l2_hidden_nodes[b];
            connection->node1 = output_node;

            output_layer->connections.emplace_back(connection);
        }
    }

    AddLayer(output_layer);
}

NeuralNetwork::~NeuralNetwork() {
    for (Node* input_node : input_nodes) {
        delete input_node;
    }

    for (Node* hidden_node : hidden_nodes) {
        delete hidden_node;
    }

    for (Node* output_node : output_nodes) {
        delete output_node;
    }

    for (Layer* layer : layers) {
        for (auto it = layer->connections.begin(); it != layer->connections.end();) {
            delete *it;
            it = layer->connections.erase(it);
        }
    }
}

void NeuralNetwork::AddLayer(Layer* layer) {
    assert(current_layer < N_LAYERS);
    
    layers[current_layer++] = layer;
}

void NeuralNetwork::ClearNodes() {
    for (Node* hidden_node : hidden_nodes) {
        hidden_node->value = 0;
    }

    for (Node* output_node : output_nodes) {
        output_node->value = 0;
    }
}

std::array<float, N_OUTPUT_NODES> NeuralNetwork::GetOutputs(const std::array<float, N_INPUT_NODES> inputs) {
    ClearNodes();

    for (int i = 0; i < N_INPUT_NODES; i++) {
        float input = inputs[i] / 255;
        input_nodes[i]->value = input;
    }

    for (int i = 0; i < N_LAYERS; i++) {
        Layer* layer = layers.at(i);
        bool hidden_layer = i < (N_LAYERS - 1);

        for (Connection* connection : layer->connections) {
            connection->node1->value += connection->node0->value * connection->weight;
        }

        for (Node* node : layer->nodes) {
            node->value += node->bias;
            
            if (hidden_layer) {
                node->value = Utility::Activation::ReLU(node->value);
            }
        }
    }

    std::array<float, N_OUTPUT_NODES> outputs;
    float sum = 0.0f;

    // softmax
    for (int i = 0; i < N_OUTPUT_NODES; i++) {
        outputs[i] = output_nodes[i]->value;
        sum += std::exp(outputs[i]);
    }

    sum = std::fmax(sum, 1e-15f);

    // return outputs as probabilities
    std::array<float, N_OUTPUT_NODES> confidences;

    for (int i = 0; i < N_OUTPUT_NODES; i++) {
        confidences[i] = std::exp(outputs[i]) / sum;
    }

    return confidences;
}

// std::array<float, N_OUTPUT_NODES> NeuralNetwork::GetConfidences(const std::array<float, N_OUTPUT_NODES> outputs) {
//     std::array<float, N_OUTPUT_NODES> confidences;

//     float sum = 0;

//     for (float output : outputs) {
//         sum += std::exp(output);
//     }

//     for (int i = 0; i < N_OUTPUT_NODES; i++) {
//         confidences[i] = std::exp(outputs[i]) / sum;
//     }

//     return confidences;
// }