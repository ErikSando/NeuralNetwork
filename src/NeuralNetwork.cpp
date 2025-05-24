#include <cmath>

#include <DataParser.h>
#include <NeuralNetwork.h>
#include <Utility.h>

NeuralNetwork::NeuralNetwork() {
    //float weight_max = std::sqrt(6 / (N_INPUT_NODES + N_OUTPUT_NODES));
    float weight_max = std::sqrt(2.0f / static_cast<float>(N_INPUT_NODES));
    float weight_min = -weight_max;

    int size = HIDDEN_LAYER_1_SIZE * N_INPUT_NODES;
    h1_weights.Resize(HIDDEN_LAYER_1_SIZE, N_INPUT_NODES);
    h1_weights.elements = Utility::RandomMatrixElements(size, weight_min, weight_max);

    size = HIDDEN_LAYER_2_SIZE * HIDDEN_LAYER_1_SIZE;
    h2_weights.Resize(HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_1_SIZE);
    h2_weights.elements = Utility::RandomMatrixElements(size, weight_min, weight_max);

    size = N_OUTPUT_NODES * HIDDEN_LAYER_2_SIZE;
    out_weights.Resize(N_OUTPUT_NODES, HIDDEN_LAYER_2_SIZE);
    out_weights.elements = Utility::RandomMatrixElements(size, weight_min, weight_max);

    h1_biases.Resize(HIDDEN_LAYER_1_SIZE, BATCH_SIZE);
    h2_biases.Resize(HIDDEN_LAYER_2_SIZE, BATCH_SIZE);
    out_biases.Resize(N_OUTPUT_NODES, BATCH_SIZE);

    h1_nodes.Resize(HIDDEN_LAYER_1_SIZE, BATCH_SIZE);
    h2_nodes.Resize(HIDDEN_LAYER_2_SIZE, BATCH_SIZE);
    out_nodes.Resize(N_OUTPUT_NODES, BATCH_SIZE);
}

NeuralNetwork::~NeuralNetwork() {
    // don't need this anymore, now that I'm not using heap allocated pointers
}

std::array<float, N_OUTPUT_NODES * BATCH_SIZE> NeuralNetwork::GetOutputs(const std::array<uint8_t, N_INPUT_NODES * BATCH_SIZE> inputs) {
    Matrix input_matrix{ /*.rows = N_INPUT_NODES, .columns = BATCH_SIZE*/ };
    input_matrix.Resize(N_INPUT_NODES, BATCH_SIZE);

    for (int i = 0; i < N_INPUT_NODES * BATCH_SIZE; i++) {
        float input = inputs[i] / 255;
        input_matrix.elements[i] = input;
    }

    h1_nodes = h1_weights * input_matrix + h1_biases;

    for (int i = 0; i < HIDDEN_LAYER_1_SIZE * BATCH_SIZE; i++) {
        assert(i < h1_nodes.elements.size());
        h1_nodes.elements[i] = Utility::Activation::LeakyReLU(h1_nodes.elements[i]);
    }

    h2_nodes = h2_weights * h1_nodes + h2_biases;

    for (int i = 0; i < HIDDEN_LAYER_2_SIZE * BATCH_SIZE; i++) {
        assert(i < h2_nodes.elements.size());
        h2_nodes.elements[i] = Utility::Activation::LeakyReLU(h2_nodes.elements[i]);
    }

    out_nodes = out_weights * h2_nodes + out_biases;

    std::array<float, N_OUTPUT_NODES * BATCH_SIZE> probabilities;

    for (int j = 0; j < BATCH_SIZE; j++) {
        float sum = 0.0f;

        // softmax
        for (int i = 0; i < N_OUTPUT_NODES; i++) {
            sum += std::exp(out_nodes.elements[i + j * N_OUTPUT_NODES]);
        }
    
        sum = std::fmax(sum, 1e-15f);

        for (int i = 0; i < N_OUTPUT_NODES; i++) {
            probabilities[i + j * N_OUTPUT_NODES] = std::exp(out_nodes.elements[i + j * N_OUTPUT_NODES]) / sum;
        }
    }

    return probabilities;
}

// std::array<float, N_OUTPUT_NODES> NeuralNetwork::GetOutputs(const std::array<uint8_t, N_INPUT_NODES> inputs, int n_batches) {
    // if (n_batches <= 0) {
    //     n_batches = batch_size;
    // }

    // n_batches = 1; // for now

    // Matrix input_matrix{ .rows = N_INPUT_NODES, .columns = n_batches };
    // input_matrix.elements.resize(N_INPUT_NODES * n_batches);

    // assert(n_batches >= 1);
    // assert(inputs.size() == N_INPUT_NODES * n_batches);

    // Matrix batched_h1_biases{ .rows = HIDDEN_LAYER_1_SIZE, .columns = n_batches };
    // Matrix batched_h2_biases{ .rows = HIDDEN_LAYER_1_SIZE, .columns = n_batches };
    // Matrix batched_out_biases{ .rows = HIDDEN_LAYER_1_SIZE, .columns = n_batches };

    // batched_h1_biases.elements.resize(HIDDEN_LAYER_1_SIZE * n_batches);
    // batched_h2_biases.elements.resize(HIDDEN_LAYER_2_SIZE * n_batches);
    // batched_out_biases.elements.resize(N_OUTPUT_NODES * n_batches);

    // for (int c = 0; c < n_batches; c++) {
    //     for (int r = 0; r < HIDDEN_LAYER_1_SIZE; r++) {
    //         batched_h1_biases.elements[r * n_batches + c] = h1_biases.elements[r];
    //     }

    //     for (int r = 0; r < HIDDEN_LAYER_2_SIZE; r++) {
    //         batched_h2_biases.elements[r * n_batches + c] = h2_biases.elements[r];
    //     }

    //     for (int r = 0; r < N_OUTPUT_NODES; r++) {
    //         batched_out_biases.elements[r * n_batches + c] = out_biases.elements[r];
    //     }
    // }

    // for (int i = 0; i < N_INPUT_NODES * n_batches; i++) {
    //     float input = inputs[i] / 255;
    //     input_matrix.elements[i] = input;
    // }

    // h1_nodes = h1_weights * input_matrix + h1_biases;

    // for (int i = 0; i < HIDDEN_LAYER_1_SIZE * n_batches; i++) {
    //     assert(i < h1_nodes.elements.size());
    //     h1_nodes.elements[i] = Utility::Activation::LeakyReLU(h1_nodes.elements[i]);
    // }

    // h2_nodes = h2_weights * h1_nodes + h2_biases;

    // for (int i = 0; i < HIDDEN_LAYER_2_SIZE * n_batches; i++) {
    //     assert(i < h2_nodes.elements.size());
    //     h2_nodes.elements[i] = Utility::Activation::LeakyReLU(h2_nodes.elements[i]);
    // }

    // out_nodes = out_weights * h2_nodes + out_biases;

    // std::array<float, N_OUTPUT_NODES> outputs;
    // std::array<float, N_OUTPUT_NODES> probabilities;
    // float sum = 0.0f;

    // // softmax
    // for (int i = 0; i < N_OUTPUT_NODES; i++) {
    //     outputs[i] = out_nodes.elements[i];
    //     sum += std::exp(outputs[i]);
    // }

    // sum = std::fmax(sum, 1e-15f);

    // for (int i = 0; i < N_OUTPUT_NODES; i++) {
    //     probabilities[i] = std::exp(outputs[i]) / sum;
    // }

    // return probabilities;
// }

// std::array<float, N_OUTPUT_NODES> NeuralNetwork::GetOutputs(const std::array<float, N_INPUT_NODES> inputs) {
//     Matrix input_matrix{ .rows = N_INPUT_NODES, .columns = 1 };
//     input_matrix.elements.resize(N_INPUT_NODES);

//     for (int i = 0; i < N_INPUT_NODES; i++) {
//         float input = inputs[i] / 255;
//         input_matrix.elements[i] = input;
//     }

//     h1_nodes = h1_weights * input_matrix + h1_biases;

//     for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
//         assert(i < h1_nodes.elements.size());
//         h1_nodes.elements[i] = Utility::Activation::LeakyReLU(h1_nodes.elements[i]);
//     }

//     h2_nodes = h2_weights * h1_nodes + h2_biases;

//     for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
//         assert(i < h2_nodes.elements.size());
//         h2_nodes.elements[i] = Utility::Activation::LeakyReLU(h2_nodes.elements[i]);
//     }

//     out_nodes = out_weights * h2_nodes + out_biases;

//     std::array<float, N_OUTPUT_NODES> outputs;
//     std::array<float, N_OUTPUT_NODES> probabilities;
//     float sum = 0.0f;

//     // softmax
//     for (int i = 0; i < N_OUTPUT_NODES; i++) {
//         outputs[i] = out_nodes.elements[i];
//         sum += std::exp(outputs[i]);
//     }

//     sum = std::fmax(sum, 1e-15f);

//     for (int i = 0; i < N_OUTPUT_NODES; i++) {
//         probabilities[i] = std::exp(outputs[i]) / sum;
//     }

//     return probabilities;
// }