#include <iostream>
#include <fstream>

#include <NeuralNetwork.h>
#include <Utility.h>

bool NeuralNetwork::SaveModel(const std::string& savepath) {
    std::ofstream output(savepath, std::ios::trunc);

    int num_connections[N_LAYERS] = {
        HIDDEN_LAYER_1_SIZE * N_INPUT_NODES,
        HIDDEN_LAYER_1_SIZE * HIDDEN_LAYER_2_SIZE,
        HIDDEN_LAYER_2_SIZE * N_OUTPUT_NODES
    };

    // num_connections[0] = N_NODES_PER_HIDDEN_LAYER * N_INPUT_NODES;
    // num_connections[N_LAYERS - 1] = N_NODES_PER_HIDDEN_LAYER * N_OUTPUT_NODES;

    // int total_connections = num_connections[0] + num_connections[N_LAYERS - 1];

    // for (int i = 1; i < N_LAYERS - 1; i++) {
    //     num_connections[i] = N_NODES_PER_HIDDEN_LAYER * N_NODES_PER_HIDDEN_LAYER;
    //     total_connections += num_connections[i];
    // }

    int total_connections = 0;

    for (int i = 0; i < N_LAYERS; i++) {
        total_connections += num_connections[i];
    }

    int layer = 0;
    int connection = 0;

    if (output.is_open()) {
        while (layer < N_LAYERS) {
            output << layers[layer]->connections.at(connection)->weight << " ";
            connection++;

            if (connection >= num_connections[layer]) {
                connection = 0;
                layer++;
            }
        }

        output << "\n";

        for (int i = 0; i < N_HIDDEN_NODES; i++) {
            output << hidden_nodes[i]->bias << " ";
        }

        output.close();

        return true;
    }
    else {
        std::cerr << "Failed to save model, could not open " << savepath << std::endl;
        return false;
    }
}

void NeuralNetwork::UploadModel(const std::string& savepath) {
    std::string raw_text = Utility::ReadFile(savepath);

    int i = 0;
    char c = raw_text.at(i);
    
    std::string current_val = "";

    int num_connections[N_LAYERS] = {
        HIDDEN_LAYER_1_SIZE * N_INPUT_NODES,
        HIDDEN_LAYER_1_SIZE * HIDDEN_LAYER_2_SIZE,
        HIDDEN_LAYER_2_SIZE * N_OUTPUT_NODES
    };

    // num_connections[0] = N_NODES_PER_HIDDEN_LAYER * N_INPUT_NODES;
    // num_connections[N_LAYERS - 1] = N_NODES_PER_HIDDEN_LAYER * N_OUTPUT_NODES;

    // int total_connections = num_connections[0] + num_connections[N_LAYERS - 1];

    // for (int i = 1; i < N_LAYERS - 1; i++) {
    //     num_connections[i] = N_NODES_PER_HIDDEN_LAYER * N_NODES_PER_HIDDEN_LAYER;
    //     total_connections += num_connections[i];
    // }

    int total_connections = 0;

    for (int i = 0; i < N_LAYERS; i++) {
        total_connections += num_connections[i];
    }

    int n_weights = 0;
    int layer = 0;
    int connection = 0;

    // upload weights data
    while (c != '\n' && i < raw_text.size()) {
        c = raw_text.at(i++);

        while(std::isdigit(c) || c == '.' || c == '-') {
            current_val += c;
            c = raw_text.at(i++);
        }

        if (current_val != "") {
            if (connection >= num_connections[layer]) {
                connection = 0;
                layer++;

                if (layer >= N_LAYERS) {
                    std::cout << current_val << std::endl;
                    std::cerr << "Too many weights, reached limit of " << n_weights << ". Ignoring remaining weights." << std::endl;
                    break;
                }
            }

            layers[layer]->connections.at(connection)->weight = std::stof(current_val);

            current_val = "";
            connection++;
            n_weights++;
        }
    }

    if (n_weights < total_connections) {
        std::cerr << "Too few weights, expected: " << total_connections << ", got: " << n_weights << ". Skipping remaining weights." << std::endl;
    }

    int n_biases = 0;
    int hidden_node = 0;

    // upload biases data
    while (i < raw_text.size()) {
        c = raw_text.at(i++);

        while(std::isdigit(c) || c == '.' || c == '-') {
            current_val += c;
            c = raw_text.at(i++);
        }

        if (current_val != "") {
            if (hidden_node >= N_HIDDEN_NODES) {
                std::cout << current_val << std::endl;
                std::cerr << "Too many biases, reached limit of " << N_HIDDEN_NODES << ". Ignoring remaining biases." << std::endl;
                break;
            }
            
            hidden_nodes[hidden_node]->bias = std::stof(current_val);

            current_val = "";
            hidden_node++;
            n_biases++;
        }
    }

    if (n_biases < N_HIDDEN_NODES) {
        std::cerr << "Too few biases, expected: " << N_HIDDEN_NODES << ", got: " << n_biases << ". Remaining biases will be left at zero." << std::endl;
    }
}