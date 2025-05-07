#include <algorithm>

#include <DataParser.h>
#include <NeuralNetwork.h>
#include <Utility.h>

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

        assert(current_row <= TRAINING_ROWS);

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
                    connection->weight -= learning_rate * (delta * node0_val + reg_str * connection->weight);
                    //connection->weight -= learning_rate * delta * node0_val;
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
                        connection->weight -= learning_rate * (delta * node0_val + reg_str * connection->weight);
                        //connection->weight -= learning_rate * delta * node0_val;
                        connection->node1->bias -= learning_rate * delta;
                        break;
                    }
                }
            }
        }
    }
}