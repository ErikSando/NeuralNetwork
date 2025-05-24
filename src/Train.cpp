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
            std::copy(image_data[i]->pixels.begin(), image_data[i]->pixels.end(), inputs.begin());
            digits[i] = image_data[i]->digit;

            //std::cout << image_data[i]->digit << std::endl;

            delete image_data[i];
        }

        std::array<float, N_OUTPUT_NODES * BATCH_SIZE> outputs = GetOutputs(inputs);
        std::array<uint8_t, N_OUTPUT_NODES * BATCH_SIZE> targets = Utility::GetBatchedTrueOutputs(digits);

        std::array<float, N_OUTPUT_NODES> output_deltas;

        // L: loss
        // y: output node value (softmax)
        // n: hidden node value (leaky ReLU)
        // w: weight

        for (int j = 0; j < N_OUTPUT_NODES; j++) {
            output_deltas[j] = outputs[j] - static_cast<float>(targets[j]); // dL / dy
        }

        std::array<float, HIDDEN_LAYER_2_SIZE> h2_deltas{};
        std::array<float, HIDDEN_LAYER_1_SIZE> h1_deltas{};

        for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
            float activation_derivative = h2_nodes.elements[i] > 0 ? 1.0f : 0.01f;
            float delta_sum = 0.0f;

            for (int j = 0; j < N_OUTPUT_NODES; j++) {
                delta_sum += output_deltas[j] * out_weights.elements[i + j * HIDDEN_LAYER_2_SIZE];
            }

            h2_deltas[i] += delta_sum * activation_derivative;
        }

        for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
            float activation_derivative = h1_nodes.elements[i] > 0 ? 1.0f : 0.01f;
            float delta_sum = 0.0f;

            for (int j = 0; j < HIDDEN_LAYER_2_SIZE; j++) {
                delta_sum += h2_deltas[j] * h2_weights.elements[i + j * HIDDEN_LAYER_1_SIZE];
            }

            h1_deltas[i] += delta_sum * activation_derivative;
        }

        // might be able to speed up the stuff below with matrix multiplication

        for (int i = 0; i < N_OUTPUT_NODES; i++) {
            float delta = std::clamp(output_deltas[i], -1.0f, 1.0f);
            //float delta = output_deltas[i];
            int offset = i * HIDDEN_LAYER_2_SIZE;

            for (int j = 0; j < HIDDEN_LAYER_2_SIZE; j++) {
                out_weight_gradients[j + offset] -= learning_rate * (delta * h2_nodes.elements[j] + reg_str * out_weights.elements[j + offset]);
            }

            out_bias_gradients[i] -= learning_rate * delta;
        }

        for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
            float delta = std::clamp(h2_deltas[i], -1.0f, 1.0f);
            //float delta = h2_deltas[i];
            int offset = i * HIDDEN_LAYER_1_SIZE;

            for (int j = 0; j < HIDDEN_LAYER_1_SIZE; j++) {
                h2_weight_gradients[j + offset] -= learning_rate * (delta * h1_nodes.elements[j] + reg_str * h2_weights.elements[j + offset]);
            }

            h2_bias_gradients[i] -= learning_rate * delta;
        }

        for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
            float delta = std::clamp(h1_deltas[i], -1.0f, 1.0f);
            //float delta = h1_deltas[i];
            int offset = i * N_INPUT_NODES;

            for (int j = 0; j < N_INPUT_NODES; j++) {
                h1_weight_gradients[j + offset] -= learning_rate * (delta * static_cast<float>(inputs[j]) / 255 + reg_str * h1_weights.elements[j + offset]);
            }

            h1_bias_gradients[i] -= learning_rate * delta;
        }

        for (int i = 0; i < N_OUTPUT_NODES; i++) {
            int offset = i * HIDDEN_LAYER_2_SIZE;

            for (int j = 0; j < HIDDEN_LAYER_2_SIZE; j++) {
                out_weights.elements[j + offset] -= out_weight_gradients[j + offset];
            }
    
            out_biases.elements[i] -= out_bias_gradients[i];
        }
    
        for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
            int offset = i * HIDDEN_LAYER_1_SIZE;

            for (int j = 0; j < HIDDEN_LAYER_1_SIZE; j++) {
                h2_weights.elements[j + offset] -= h2_weight_gradients[j + offset];
            }
    
            h2_biases.elements[i] -= h2_bias_gradients[i];
        }
    
        for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
            int offset = i * N_INPUT_NODES;

            for (int j = 0; j < N_INPUT_NODES; j++) {
                h1_weights.elements[j + offset] -= h1_weight_gradients[j + offset];
            }
    
            h1_biases.elements[i] -= h1_bias_gradients[i];
        }
    }
}