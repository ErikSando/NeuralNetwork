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
    h1_weights.elements = Matrix::RandomMatrixElements(size, weight_min, weight_max);

    size = HIDDEN_LAYER_2_SIZE * HIDDEN_LAYER_1_SIZE;
    h2_weights.Resize(HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_1_SIZE);
    h2_weights.elements = Matrix::RandomMatrixElements(size, weight_min, weight_max);

    size = N_OUTPUT_NODES * HIDDEN_LAYER_2_SIZE;
    out_weights.Resize(N_OUTPUT_NODES, HIDDEN_LAYER_2_SIZE);
    out_weights.elements = Matrix::RandomMatrixElements(size, weight_min, weight_max);

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

    //h1_nodes = h1_weights * input_matrix + h1_biases;

    // testing performance different with operator overloading and this
    Matrix::MatrixMultiply(h1_weights, input_matrix, h1_nodes);
    Matrix::MatrixAdd(h1_nodes, h1_biases, h1_nodes);

    for (int i = 0; i < HIDDEN_LAYER_1_SIZE * BATCH_SIZE; i++) {
        h1_nodes.elements[i] = Utility::Activation::LeakyReLU(h1_nodes.elements[i]);
    }

    //h2_nodes = h2_weights * h1_nodes + h2_biases;

    Matrix::MatrixMultiply(h2_weights, h1_nodes, h2_nodes);
    Matrix::MatrixAdd(h2_nodes, h2_biases, h2_nodes);

    for (int i = 0; i < HIDDEN_LAYER_2_SIZE * BATCH_SIZE; i++) {
        h2_nodes.elements[i] = Utility::Activation::LeakyReLU(h2_nodes.elements[i]);
    }

    //out_nodes = out_weights * h2_nodes + out_biases;

    Matrix::MatrixMultiply(out_weights, h2_nodes, out_nodes);
    Matrix::MatrixAdd(out_nodes, out_biases, out_nodes);

    std::array<float, N_OUTPUT_NODES * BATCH_SIZE> probabilities;

    for (int j = 0; j < BATCH_SIZE; j++) {
        float sum = 0.0f;

        int start = j * N_OUTPUT_NODES;

        // softmax
        for (int i = 0; i < N_OUTPUT_NODES; i++) {
            sum += std::exp(out_nodes.elements[i + start]);
        }
    
        sum = std::fmax(sum, 1e-15f);

        for (int i = 0; i < N_OUTPUT_NODES; i++) {
            probabilities[i + start] = std::exp(out_nodes.elements[i + start]) / sum;
        }
    }

    return probabilities;
}