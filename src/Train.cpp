#include <algorithm>
#include <cmath>
#include <string.h>

#include <DataParser.h>
#include <NeuralNetwork.h>
#include <Utility.h>

void NeuralNetwork::SetLearningRate(const float lr) {
    learning_rate = lr;
}

// void NeuralNetwork::SetBatchSize(const int bs) {
//     batch_size = bs;
// }

void NeuralNetwork::Train(const int batches, const std::string& trainpath, const bool newpath) {
    if (newpath) current_row = 1;

    for (int _ = 0; _ < batches; _++) {
        std::array<float, N_OUTPUT_NODES * HIDDEN_LAYER_2_SIZE> out_weight_gradients{};
        std::array<float, HIDDEN_LAYER_2_SIZE * HIDDEN_LAYER_1_SIZE> h2_weight_gradients{};
        std::array<float, HIDDEN_LAYER_1_SIZE * N_INPUT_NODES> h1_weight_gradients{};

        std::array<float, N_OUTPUT_NODES> out_bias_gradients{};
        std::array<float, HIDDEN_LAYER_2_SIZE> h2_bias_gradients{};
        std::array<float, HIDDEN_LAYER_1_SIZE> h1_bias_gradients{};

        std::array<ImageData*, BATCH_SIZE> image_data = DataParser::GetBatchedTrainingData(current_row, trainpath);
        std::array<int, BATCH_SIZE> digits;
        std::array<uint8_t, N_INPUT_NODES * BATCH_SIZE> inputs;

        current_row = ((current_row + BATCH_SIZE - 1) % TRAINING_ROWS) + 1;
        assert(current_row <= TRAINING_ROWS);

        for (int i = 0; i < BATCH_SIZE; i++) {
            std::copy(image_data[i]->pixels.begin(), image_data[i]->pixels.end(), inputs.begin() + i * N_INPUT_NODES);
            digits[i] = image_data[i]->digit;

            //std::cout << image_data[i]->digit << std::endl;

            delete image_data[i];
        }

        std::array<float, N_OUTPUT_NODES * BATCH_SIZE> outputs = GetOutputs(inputs);
        std::array<uint8_t, N_OUTPUT_NODES * BATCH_SIZE> targets{};
        Utility::GetBatchedTrueOutputs(digits, targets);

        // L: loss
        // y: output node value (softmax)
        // n: hidden node value (leaky ReLU)
        // w: weight

        for (int b = 0; b < BATCH_SIZE; b++) {
            int inp_offset = b * N_INPUT_NODES;
            int out_offset = b * N_OUTPUT_NODES;
            int h1_offset = b * HIDDEN_LAYER_1_SIZE;
            int h2_offset = b * HIDDEN_LAYER_2_SIZE;

            std::array<float, N_OUTPUT_NODES> output_deltas;

            for (int j = 0; j < N_OUTPUT_NODES; j++) {
                output_deltas[j] = outputs[j + out_offset] - static_cast<float>(targets[j + out_offset]); // dL / dy
            }

            std::array<float, HIDDEN_LAYER_2_SIZE> h2_deltas;
            std::array<float, HIDDEN_LAYER_1_SIZE> h1_deltas;

            for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
                float activation_derivative = h2_nodes.elements[i + h2_offset] > 0 ? 1.0f : 0.01f;
                float delta_sum = 0.0f;

                for (int j = 0; j < N_OUTPUT_NODES; j++) {
                    delta_sum += output_deltas[j] * out_weights.elements[i + j * HIDDEN_LAYER_2_SIZE];
                }

                h2_deltas[i] = delta_sum * activation_derivative;
            }

            for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
                float activation_derivative = h1_nodes.elements[i + h1_offset] > 0 ? 1.0f : 0.01f;
                float delta_sum = 0.0f;

                for (int j = 0; j < HIDDEN_LAYER_2_SIZE; j++) {
                    delta_sum += h2_deltas[j] * h2_weights.elements[i + j * HIDDEN_LAYER_1_SIZE];
                }

                h1_deltas[i] = delta_sum * activation_derivative;
            }

            // might be able to speed up the stuff below with matrix multiplication
            for (int i = 0; i < N_OUTPUT_NODES; i++) {
                float delta = std::clamp(output_deltas[i], -1.0f, 1.0f);
                //float delta = output_deltas[i];
                int offset = i * HIDDEN_LAYER_2_SIZE;

                for (int j = 0; j < HIDDEN_LAYER_2_SIZE; j++) {
                    out_weight_gradients[j + offset] -= delta * h2_nodes.elements[j + h2_offset];
                }

                out_bias_gradients[i] -= delta;
            }

            for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
                float delta = std::clamp(h2_deltas[i], -1.0f, 1.0f);
                //float delta = h2_deltas[i];
                int offset = i * HIDDEN_LAYER_1_SIZE;

                for (int j = 0; j < HIDDEN_LAYER_1_SIZE; j++) {
                    h2_weight_gradients[j + offset] -= delta * h1_nodes.elements[j + h1_offset];
                }

                h2_bias_gradients[i] -= delta;
            }

            for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
                float delta = std::clamp(h1_deltas[i], -1.0f, 1.0f);
                //float delta = h1_deltas[i];
                int offset = i * N_INPUT_NODES;

                for (int j = 0; j < N_INPUT_NODES; j++) {
                    h1_weight_gradients[j + offset] -= delta * static_cast<float>(inputs[j + inp_offset]) / 255;
                }

                h1_bias_gradients[i] -= delta;
            }
        }

        for (int i = 0; i < N_OUTPUT_NODES; i++) {
            int offset = i * HIDDEN_LAYER_2_SIZE;

            for (int j = 0; j < HIDDEN_LAYER_2_SIZE; j++) {
                out_weights.elements[j + offset] -= learning_rate * (out_weight_gradients[j + offset] + reg_str * out_weights.elements[j + offset]) / BATCH_SIZE;
            }
    
            out_biases.elements[i] -= learning_rate * out_bias_gradients[i] / BATCH_SIZE;
        }
    
        for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
            int offset = i * HIDDEN_LAYER_1_SIZE;

            for (int j = 0; j < HIDDEN_LAYER_1_SIZE; j++) {
                h2_weights.elements[j + offset] -= learning_rate * (h2_weight_gradients[j + offset] + reg_str * h2_weights.elements[j + offset]) / BATCH_SIZE;
            }
    
            h2_biases.elements[i] -= learning_rate * h2_bias_gradients[i] / BATCH_SIZE;
        }
    
        for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
            int offset = i * N_INPUT_NODES;

            for (int j = 0; j < N_INPUT_NODES; j++) {
                h1_weights.elements[j + offset] -= learning_rate * (h1_weight_gradients[j + offset] + reg_str * h1_weights.elements[j + offset]) / BATCH_SIZE;
            }
    
            h1_biases.elements[i] -= learning_rate * h1_bias_gradients[i] / BATCH_SIZE;
        }
    }
}