#include <algorithm>
#include <cmath>

#include <DataParser.h>
#include <NeuralNetwork.h>
#include <Utility.h>

NeuralNetwork::NeuralNetwork() {
    for (int a = 0; a < N_INPUT_NODES; a++) {
        Node* input_node = new Node;
        input_nodes[a] = input_node;
    }

    std::vector<Node*> _hidden_nodes;

    Layer* first_hidden_layer = new Layer;

    float weight_max = 6 / std::sqrt(N_INPUT_NODES + N_OUTPUT_NODES);
    float weight_min = -weight_max;

    for (int a = 0; a < N_NODES_PER_HIDDEN_LAYER; a++) {
        Node* hidden_node = new Node;

        hidden_nodes[a] = hidden_node;
        first_hidden_layer->nodes.emplace_back(hidden_node);
        _hidden_nodes.emplace_back(hidden_node);

        // connect with every input node
        for (int b = 0; b < N_INPUT_NODES; b++) {
            Connection* connection = new Connection;
            connection->weight = Random::Float(weight_min, weight_max);
            connection->node0 = input_nodes[b];
            connection->node1 = hidden_node;

            first_hidden_layer->connections.emplace_back(connection);
        }
    }

    AddLayer(first_hidden_layer);

    std::vector<Node*> next_hidden_nodes;

    for (int a = 1; a < N_HIDDEN_LAYERS; a++) {
        Layer* hidden_layer = new Layer;

        for (int b = 0; b < N_NODES_PER_HIDDEN_LAYER; b++) {
            Node* hidden_node = new Node;

            hidden_nodes[b + a * N_NODES_PER_HIDDEN_LAYER] = hidden_node;
            hidden_layer->nodes.emplace_back(hidden_node);
            next_hidden_nodes.emplace_back(hidden_node);

            // connect with every hidden node in the previous layer
            for (int c = 0; c < N_NODES_PER_HIDDEN_LAYER; c++) {
                Connection* connection = new Connection;
                connection->weight = Random::Float(weight_min, weight_max);
                connection->node0 = _hidden_nodes.at(c);
                connection->node1 = hidden_node;

                hidden_layer->connections.emplace_back(connection);
            }
        }

        AddLayer(hidden_layer);

        _hidden_nodes = next_hidden_nodes;
    }

    Layer* output_layer = new Layer;

    for (int a = 0; a < N_OUTPUT_NODES; a++) {
        Node* output_node = new Node;

        output_nodes[a] = output_node;
        output_layer->nodes.emplace_back(output_node);

        for (int b = 0; b < N_NODES_PER_HIDDEN_LAYER; b++) {
            Connection* connection = new Connection;
            connection->weight = Random::Float(weight_min, weight_max);
            connection->node0 = _hidden_nodes.at(b);
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
                node->value = Utility::ActivationFunctions::ReLU(node->value);
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

void NeuralNetwork::SetLearningRate(const float lr) {
    learning_rate = lr;
}

void NeuralNetwork::Train(const int iterations, const std::string& trainpath, const bool newpath) {
    if (newpath) current_row = 1;

    for (int i = 0; i < iterations; i++) {
        ImageData* data = DataParser::GetRowImageData(current_row, trainpath);

        std::array<float, N_OUTPUT_NODES> outputs = GetOutputs(data->pixels);
        std::array<float, N_OUTPUT_NODES> targets = Utility::GetTrueOutputs(data->digit);

        #ifndef NDEBUG

        for (int i = 0; i < 10; i++) {
            float expected = (i == data->digit) ? 1.0f : 0.0f;
            assert(targets[i] == expected);
        }

        #endif

        delete data;

        current_row = (current_row % TRAINING_ROWS) + 1;

        std::array<float, N_OUTPUT_NODES> output_deltas;
        for (int j = 0; j < N_OUTPUT_NODES; j++) {
            output_deltas[j] = outputs[j] - targets[j];
        }

        std::vector<std::vector<float>> hidden_deltas(N_HIDDEN_LAYERS);

        for (int li = N_LAYERS - 2; li >= 0; li--) {
            Layer* layer = layers[li];
            bool is_last_hidden = (li == N_LAYERS - 2);

            int n_nodes = layer->nodes.size(); // constant, but I want to eventually support different sized hidden layers
            hidden_deltas.at(li).resize(n_nodes);

            for (int j = 0; j < n_nodes; j++) {
                Node* node = layer->nodes[j];

                float a_derivative = node->value > 0 ? 1.0f : 0.0f;
                float delta_sum = 0.0f;

                if (a_derivative == 0.0f) {
                    hidden_deltas[li][j] = 0.0f;
                    continue;
                }

                Layer* next_layer = layers[li + 1];
                for (Connection* connection : next_layer->connections) {
                    if (node == connection->node0) {
                        float delta = 0.0f;

                        if (is_last_hidden) {
                            int output_index = -1;
                        
                            for (int k = 0; k < N_OUTPUT_NODES; k++) {
                                if (output_nodes[k] == connection->node1) {
                                    output_index = k;
                                    break;
                                }
                            }

                            assert(output_index >= 0);

                            delta = output_deltas[output_index];
                        }
                        else {
                            int hidden_index = -1;

                            for (int k = 0; k < next_layer->nodes.size(); k++) {
                                if (next_layer->nodes.at(k) == connection->node1) {
                                    hidden_index = k;
                                    break;
                                }
                            }

                            assert(hidden_index >= 0);

                            delta = hidden_deltas[li + 1][hidden_index];
                        }
                        
                        delta_sum += connection->weight * delta;
                    }
                }

                hidden_deltas[li][j] = delta_sum * a_derivative;
            }
        }
    
        Layer* output_layer = layers[N_LAYERS - 1];
        for (Connection* connection : output_layer->connections) {
            float node0_val = connection->node0->value;

            for (int j = 0; j < N_OUTPUT_NODES; j++) {
                if (output_nodes[j] == connection->node1) {
                    float delta = std::clamp(output_deltas[j], -1.0f, 1.0f);
                    connection->weight -= learning_rate * delta * node0_val;
                    connection->node1->bias -= learning_rate * delta;
                    break;
                }
            }
        }

        for (int li = N_LAYERS - 2; li >= 0; li--) {
            Layer* layer = layers[li];
            
            for (Connection* connection : layer->connections) {
                float node0_val = connection->node0->value;

                for (int j = 0; j < (int) layer->nodes.size(); j++) {
                    if (layer->nodes[j] == connection->node1) {
                        float delta = std::clamp(hidden_deltas[li][j], -1.0f, 1.0f);
                        connection->weight -= learning_rate * delta * node0_val;
                        connection->node1->bias -= learning_rate * delta;
                        break;
                    }
                }
            }
        }
    }
}

TestingData* NeuralNetwork::Test(const int iterations, const std::string& testpath, const bool newpath) {
    if (newpath) c_training_row = 1;

    TestingData* t_data = new TestingData;

    int correct = 0;
    int incorret = 0;

    for (int i = 0; i < iterations; i++) {
        ImageData* data = DataParser::GetRowImageData(c_training_row, testpath);

        std::array<float, N_OUTPUT_NODES> outputs = GetOutputs(data->pixels);

        c_training_row = (c_training_row % TRAINING_ROWS) + 1;

        // repeated code, TODO: prevent repeating
        float largest = 0;
        int l_digit = -1;

        for (int i = 0; i < N_OUTPUT_NODES; i++) {
            if (outputs[i] > largest) {
                largest = outputs[i];
                l_digit = i;
            }
        }

        #ifndef NDEBUG
        
        std::cout << "Prediction: " << l_digit << " (" << data->digit << ")" << std::endl;

        #endif

        if (l_digit == data->digit) t_data->correct++;
        else t_data->incorrect++;

        delete data;
    }

    return t_data;
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