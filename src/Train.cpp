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
            //memcpy((void*) &image_data[i]->pixels, (void*) &inputs[i * BATCH_SIZE], i * sizeof(uint8_t) * N_INPUT_NODES);
            std::copy(image_data[i]->pixels.begin(), image_data[i]->pixels.end(), inputs.begin());
            digits[i] = image_data[i]->digit;

            //std::cout << image_data[i]->digit << std::endl;

            delete image_data[i];
        }

        std::array<float, N_OUTPUT_NODES * BATCH_SIZE> outputs = GetOutputs(inputs);
        std::array<uint8_t, N_OUTPUT_NODES * BATCH_SIZE> targets = Utility::GetBatchedTrueOutputs(digits);

        std::array<float, N_OUTPUT_NODES> output_deltas;

        for (int j = 0; j < N_OUTPUT_NODES; j++) {
            output_deltas[j] = outputs[j] - static_cast<float>(targets[j]);
        }

        std::array<float, HIDDEN_LAYER_2_SIZE> h2_deltas;
        std::array<float, HIDDEN_LAYER_1_SIZE> h1_deltas;

        for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
            float activation_derivative = h2_nodes.elements[i] > 0 ? 1.0f : 0.0f;

            if (activation_derivative == 0.0f) {
                h2_deltas[i] = 0.0f;
                continue;
            }

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

        for (int i = 0; i < N_OUTPUT_NODES; i++) {
            float delta = std::clamp(output_deltas[i], -1.0f, 1.0f);

            for (int j = 0; j < HIDDEN_LAYER_2_SIZE; j++) {
                out_weight_gradients[j + i * HIDDEN_LAYER_2_SIZE] -= learning_rate * (delta * h2_nodes.elements[j] + reg_str * out_weights.elements[j + i * HIDDEN_LAYER_2_SIZE]);
            }

            out_bias_gradients[i] -= learning_rate * delta;
        }

        for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
            float delta = std::clamp(h2_deltas[i], -1.0f, 1.0f);

            for (int j = 0; j < HIDDEN_LAYER_1_SIZE; j++) {
                h2_weight_gradients[j + i * HIDDEN_LAYER_1_SIZE] -= learning_rate * (delta * h1_nodes.elements[j] + reg_str * h2_weights.elements[j + i * HIDDEN_LAYER_1_SIZE]);
            }

            h2_bias_gradients[i] -= learning_rate * delta;
        }

        for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
            float delta = std::clamp(h1_deltas[i], -1.0f, 1.0f);

            for (int j = 0; j < N_INPUT_NODES; j++) {
                h1_weight_gradients[j + i * N_INPUT_NODES] -= learning_rate * (delta * inputs[j] / 255 + reg_str * h1_weights.elements[j + i * N_INPUT_NODES]);
            }

            h1_bias_gradients[i] -= learning_rate * delta;
        }

        for (int i = 0; i < N_OUTPUT_NODES; i++) {
            for (int j = 0; j < HIDDEN_LAYER_2_SIZE; j++) {
                out_weights.elements[j + i * HIDDEN_LAYER_2_SIZE] -= out_weight_gradients[j + i * HIDDEN_LAYER_2_SIZE];
            }
    
            out_biases.elements[i] -= out_bias_gradients[i];
        }
    
        for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_1_SIZE; j++) {
                h2_weights.elements[j + i * HIDDEN_LAYER_1_SIZE] -= h2_weight_gradients[j + i * HIDDEN_LAYER_1_SIZE];
            }
    
            h2_biases.elements[i] -= h2_bias_gradients[i];
        }
    
        for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
            for (int j = 0; j < N_INPUT_NODES; j++) {
                h1_weights.elements[j + i * N_INPUT_NODES] -= h1_weight_gradients[j + i * N_INPUT_NODES];
            }
    
            h1_biases.elements[i] -= h1_bias_gradients[i];
        }
    }
}

// void NeuralNetwork::Train(const int iterations, const std::string& trainpath, const bool newpath) {
//     if (newpath) current_row = 1;

//     int _iterations = std::floor(iterations / batch_size);

//     for (int _ = 0; _ < _iterations; _++) {
//         std::array<float, N_OUTPUT_NODES * HIDDEN_LAYER_2_SIZE> cumulative_out_weight_gradients{};
//         std::array<float, HIDDEN_LAYER_2_SIZE * HIDDEN_LAYER_1_SIZE> cumulative_h2_weight_gradients{};
//         std::array<float, HIDDEN_LAYER_1_SIZE * N_INPUT_NODES> cumulative_h1_weight_gradients{};

//         std::array<float, N_OUTPUT_NODES> cumulative_out_bias_gradients{};
//         std::array<float, HIDDEN_LAYER_2_SIZE> cumulative_h2_bias_gradients{};
//         std::array<float, HIDDEN_LAYER_1_SIZE> cumulative_h1_bias_gradients{};

//         std::vector<ImageData*> image_data = DataParser::GetRowsImageData(current_row, batch_size, trainpath);

//         for (int b = 0; b < batch_size; b++) {
//             ImageData* data = image_data[b];

//             std::array<float, N_OUTPUT_NODES> outputs = GetOutputs(data->pixels);
//             std::array<uint8_t, N_OUTPUT_NODES> targets = Utility::GetTrueOutputs(data->digit);

//             std::array<uint8_t, N_INPUT_NODES> inputs = data->pixels;

//             delete data;

//             current_row = (current_row % TRAINING_ROWS) + 1;
//             assert(current_row <= TRAINING_ROWS);

//             std::array<float, N_OUTPUT_NODES> output_deltas;

//             for (int j = 0; j < N_OUTPUT_NODES; j++) {
//                 output_deltas[j] = outputs[j] - static_cast<float>(targets[j]);
//             }

//             std::vector<float> h2_deltas(HIDDEN_LAYER_2_SIZE);
//             std::vector<float> h1_deltas(HIDDEN_LAYER_1_SIZE);

//             for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
//                 float activation_derivative = h2_nodes.elements[i] > 0 ? 1.0f : 0.0f;

//                 if (activation_derivative == 0.0f) {
//                     h2_deltas[i] = 0.0f;
//                     continue;
//                 }

//                 float delta_sum = 0.0f;

//                 for (int j = 0; j < N_OUTPUT_NODES; j++) {
//                     delta_sum += output_deltas[j] * out_weights.elements[i + j * HIDDEN_LAYER_2_SIZE];
//                 }

//                 h2_deltas[i] += delta_sum * activation_derivative;
//             }

//             for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
//                 float activation_derivative = h1_nodes.elements[i] > 0 ? 1.0f : 0.01f;

//                 // if (activation_derivative == 0.0f) {
//                 //     h1_deltas[i] = 0.0f;
//                 //     continue;
//                 // }

//                 float delta_sum = 0.0f;

//                 for (int j = 0; j < HIDDEN_LAYER_2_SIZE; j++) {
//                     delta_sum += h2_deltas[j] * h2_weights.elements[i + j * HIDDEN_LAYER_1_SIZE];
//                 }

//                 h1_deltas[i] += delta_sum * activation_derivative;
//             }

//             for (int i = 0; i < N_OUTPUT_NODES; i++) {
//                 float delta = std::clamp(output_deltas[i], -1.0f, 1.0f);

//                 for (int j = 0; j < HIDDEN_LAYER_2_SIZE; j++) {
//                     //out_weights.elements[j + i * HIDDEN_LAYER_2_SIZE] -= learning_rate * (delta * h2_nodes.elements[j] + reg_str * out_weights.elements[j + i * HIDDEN_LAYER_2_SIZE]);
//                     cumulative_out_weight_gradients[j + i * HIDDEN_LAYER_2_SIZE] -= learning_rate * (delta * h2_nodes.elements[j] + reg_str * out_weights.elements[j + i * HIDDEN_LAYER_2_SIZE]);
//                 }

//                 //out_biases.elements[i] -= learning_rate * delta;
//                 cumulative_out_bias_gradients[i] -= learning_rate * delta;
//             }

//             for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
//                 float delta = std::clamp(h2_deltas[i], -1.0f, 1.0f);

//                 for (int j = 0; j < HIDDEN_LAYER_1_SIZE; j++) {
//                     //h2_weights.elements[j + i * HIDDEN_LAYER_1_SIZE] -= learning_rate * (delta * h1_nodes.elements[j] + reg_str * h2_weights.elements[j + i * HIDDEN_LAYER_1_SIZE]);
//                     cumulative_h2_weight_gradients[j + i * HIDDEN_LAYER_1_SIZE] -= learning_rate * (delta * h1_nodes.elements[j] + reg_str * h2_weights.elements[j + i * HIDDEN_LAYER_1_SIZE]);
//                 }

//                 //h2_biases.elements[i] -= learning_rate * delta;
//                 cumulative_h2_bias_gradients[i] -= learning_rate * delta;
//             }

//             for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
//                 float delta = std::clamp(h1_deltas[i], -1.0f, 1.0f);

//                 for (int j = 0; j < N_INPUT_NODES; j++) {
//                     //h1_weights.elements[j + i * N_INPUT_NODES] -= learning_rate * (delta * inputs[j] / 255 + reg_str * h1_weights.elements[j + i * N_INPUT_NODES]);
//                     cumulative_h1_weight_gradients[j + i * N_INPUT_NODES] -= learning_rate * (delta * inputs[j] / 255 + reg_str * h1_weights.elements[j + i * N_INPUT_NODES]);
//                 }

//                 //h1_biases.elements[i] -= learning_rate * delta;
//                 cumulative_h1_bias_gradients[i] -= learning_rate * delta;
//             }
//         }

//         for (int i = 0; i < N_OUTPUT_NODES; i++) {
//             for (int j = 0; j < HIDDEN_LAYER_2_SIZE; j++) {
//                 out_weights.elements[j + i * HIDDEN_LAYER_2_SIZE] -= cumulative_out_weight_gradients[j + i * HIDDEN_LAYER_2_SIZE];
//             }

//             out_biases.elements[i] -= cumulative_out_bias_gradients[i];
//         }

//         for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
//             for (int j = 0; j < HIDDEN_LAYER_1_SIZE; j++) {
//                 h2_weights.elements[j + i * HIDDEN_LAYER_1_SIZE] -= cumulative_h2_weight_gradients[j + i * HIDDEN_LAYER_1_SIZE];
//             }

//             h2_biases.elements[i] -= cumulative_h2_bias_gradients[i];
//         }

//         for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
//             for (int j = 0; j < N_INPUT_NODES; j++) {
//                 h1_weights.elements[j + i * N_INPUT_NODES] -= cumulative_h1_weight_gradients[j + i * N_INPUT_NODES];
//             }

//             h1_biases.elements[i] -= cumulative_h1_bias_gradients[i];
//         }
//     }
// }