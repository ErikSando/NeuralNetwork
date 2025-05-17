#include <cmath>

#include <DataParser.h>
#include <NeuralNetwork.h>
#include <Utility.h>

NeuralNetwork::NeuralNetwork() {
    //float weight_max = std::sqrt(6 / (N_INPUT_NODES + N_OUTPUT_NODES));
    float weight_max = std::sqrt(2 / N_INPUT_NODES);
    float weight_min = -weight_max;

    h1_weights.rows = HIDDEN_LAYER_1_SIZE;
    h1_weights.columns = N_INPUT_NODES;
    int size = HIDDEN_LAYER_1_SIZE * N_INPUT_NODES;
    h1_weights.elements.resize(size);
    h1_weights.elements = Utility::RandomMatrixElements(size, weight_min, weight_max);

    h2_weights.rows = HIDDEN_LAYER_2_SIZE;
    h2_weights.columns = HIDDEN_LAYER_1_SIZE;
    size = HIDDEN_LAYER_2_SIZE * HIDDEN_LAYER_1_SIZE;
    h2_weights.elements.resize(size);
    h2_weights.elements = Utility::RandomMatrixElements(size, weight_min, weight_max);

    out_weights.rows = N_OUTPUT_NODES;
    out_weights.columns = HIDDEN_LAYER_2_SIZE;
    size = N_OUTPUT_NODES * HIDDEN_LAYER_2_SIZE;
    out_weights.elements.resize(size);
    out_weights.elements = Utility::RandomMatrixElements(size, weight_min, weight_max);

    h1_biases.rows = HIDDEN_LAYER_1_SIZE;
    h1_biases.columns = 1;
    h1_biases.elements.resize(HIDDEN_LAYER_1_SIZE, 0.0f);

    h2_biases.rows = HIDDEN_LAYER_2_SIZE;
    h2_biases.columns = 1;
    h2_biases.elements.resize(HIDDEN_LAYER_2_SIZE, 0.0f);

    out_biases.rows = N_OUTPUT_NODES;
    out_biases.columns = 1;
    out_biases.elements.resize(N_OUTPUT_NODES, 0.0f);
}

NeuralNetwork::~NeuralNetwork() {
    // don't need this anymore, now that I'm not using heap allocated pointers
}

std::array<float, N_OUTPUT_NODES> NeuralNetwork::GetOutputs(const std::array<uint8_t, N_INPUT_NODES> inputs, int n_batches) {
    if (n_batches <= 0) {
        n_batches = batch_size;
    }

    n_batches = 1; // for now

    Matrix input_matrix{ .rows = N_INPUT_NODES, .columns = n_batches };
    input_matrix.elements.resize(N_INPUT_NODES);

    assert(n_batches >= 1);
    assert(inputs.size() == N_INPUT_NODES * n_batches);

    Matrix batched_h1_biases{ .rows = HIDDEN_LAYER_1_SIZE, .columns = n_batches };
    Matrix batched_h2_biases{ .rows = HIDDEN_LAYER_1_SIZE, .columns = n_batches };
    Matrix batched_out_biases{ .rows = HIDDEN_LAYER_1_SIZE, .columns = n_batches };

    batched_h1_biases.elements.resize(HIDDEN_LAYER_1_SIZE * n_batches);
    batched_h2_biases.elements.resize(HIDDEN_LAYER_2_SIZE * n_batches);
    batched_out_biases.elements.resize(N_OUTPUT_NODES * n_batches);

    for (int c = 0; c < n_batches; c++) {
        for (int r = 0; r < HIDDEN_LAYER_1_SIZE; r++) {
            batched_h1_biases.elements[r * n_batches + c] = h1_biases.elements[r];
        }

        for (int r = 0; r < HIDDEN_LAYER_2_SIZE; r++) {
            batched_h2_biases.elements[r * n_batches + c] = h2_biases.elements[r];
        }

        for (int r = 0; r < N_OUTPUT_NODES; r++) {
            batched_out_biases.elements[r * n_batches + c] = out_biases.elements[r];
        }
    }

    for (int i = 0; i < N_INPUT_NODES * n_batches; i++) {
        float input = inputs[i] / 255;
        input_matrix.elements[i] = input;
    }

    h1_nodes = h1_weights * input_matrix + h1_biases;

    for (int i = 0; i < HIDDEN_LAYER_1_SIZE * n_batches; i++) {
        assert(i < h1_nodes.elements.size());
        h1_nodes.elements[i] = Utility::Activation::LeakyReLU(h1_nodes.elements[i]);
    }

    h2_nodes = h2_weights * h1_nodes + h2_biases;

    for (int i = 0; i < HIDDEN_LAYER_2_SIZE * n_batches; i++) {
        assert(i < h2_nodes.elements.size());
        h2_nodes.elements[i] = Utility::Activation::LeakyReLU(h2_nodes.elements[i]);
    }

    out_nodes = out_weights * h2_nodes + out_biases;

    std::array<float, N_OUTPUT_NODES> outputs;
    std::array<float, N_OUTPUT_NODES> probabilities;
    float sum = 0.0f;

    // softmax
    for (int i = 0; i < N_OUTPUT_NODES; i++) {
        outputs[i] = out_nodes.elements[i];
        sum += std::exp(outputs[i]);
    }

    sum = std::fmax(sum, 1e-15f);

    for (int i = 0; i < N_OUTPUT_NODES; i++) {
        probabilities[i] = std::exp(outputs[i]) / sum;
    }

    return probabilities;
}

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