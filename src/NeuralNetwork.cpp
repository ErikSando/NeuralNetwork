#include <cmath>

#include <DataParser.h>
#include <NeuralNetwork.h>
#include <Utility.h>

NeuralNetwork::NeuralNetwork() {
    float weight_max = std::sqrt(6 / (N_INPUT_NODES + N_OUTPUT_NODES));
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
}

NeuralNetwork::~NeuralNetwork() {
    
}

std::array<float, N_OUTPUT_NODES> NeuralNetwork::GetOutputs(const std::array<float, N_INPUT_NODES> inputs) {
    // Matrix* input_matrix = new Matrix;
    // input_matrix->elements.resize(N_INPUT_NODES);
    // input_matrix->rows = N_INPUT_NODES;
    // input_matrix->columns = 1;

    Matrix input_matrix;
    input_matrix.elements.resize(N_INPUT_NODES);
    input_matrix.rows = N_INPUT_NODES;
    input_matrix.columns = 1;

    for (int i = 0; i < N_INPUT_NODES; i++) {
        float input = inputs[i] / 25;
        input_matrix.elements[i] = input;
    }

    // Matrix* h1_nodes = Utility::MatrixMultiply(&h1_weights, input_matrix);
    Matrix h1_nodes = h1_weights * input_matrix + h1_biases;

    for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
        // assert(i < h1_nodes->elements.size());

        // h1_nodes->elements[i] = Utility::Activation::ReLU(h1_nodes->elements[i]);

        assert(i < h1_nodes.elements.size());

        h1_nodes.elements[i] = Utility::Activation::ReLU(h1_nodes.elements[i]);
    }

    // Matrix* h2_nodes = Utility::MatrixMultiply(&h2_weights, h1_nodes);
    Matrix h2_nodes = h2_weights * h1_nodes + h2_biases;

    for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
        // assert(i < h2_nodes->elements.size());
        
        // h2_nodes->elements[i] = Utility::Activation::ReLU(h2_nodes->elements[i]);

        assert(i < h2_nodes.elements.size());
        
        h2_nodes.elements[i] = Utility::Activation::ReLU(h2_nodes.elements[i]);
    }

    // Matrix* out_nodes = Utility::MatrixMultiply(&out_weights, h2_nodes);
    Matrix out_nodes = out_weights * h2_nodes;

    std::array<float, N_OUTPUT_NODES> outputs;
    std::array<float, N_OUTPUT_NODES> probabilities;
    float sum = 0.0f;

    // softmax
    for (int i = 0; i < N_OUTPUT_NODES; i++) {
        //outputs[i] = out_nodes->elements[i];
        outputs[i] = out_nodes.elements[i];
        sum += std::exp(outputs[i]);
    }

    sum = std::fmax(sum, 1e-15f);

    for (int i = 0; i < N_OUTPUT_NODES; i++) {
        probabilities[i] = std::exp(outputs[i]) / sum;
    }

    // delete input_matrix;
    // delete h1_nodes;
    // delete out_nodes;

    return probabilities;
}