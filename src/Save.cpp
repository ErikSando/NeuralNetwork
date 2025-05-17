#include <iostream>
#include <fstream>

#include <NeuralNetwork.h>
#include <Utility.h>

bool NeuralNetwork::SaveModel(const std::string& savepath) {
    std::ofstream output(savepath, std::ios::trunc);

    if (!output.is_open()) {
        std::cerr << "Failed to open file: " << savepath << std::endl;
        return false;
    }

    for (int i = 0; i < h1_weights.elements.size(); i++) {
        output << h1_weights.elements[i] << " ";
    }

    output << "\n";

    for (int i = 0; i < h2_weights.elements.size(); i++) {
        output << h2_weights.elements[i] << " ";
    }

    output << "\n";

    for (int i = 0; i < out_weights.elements.size(); i++) {
        output << out_weights.elements[i] << " ";
    }

    output << "\n";

    for (int i = 0; i < h1_biases.elements.size(); i++) {
        output << h1_biases.elements[i] << " ";
    }

    output << "\n";

    for (int i = 0; i < h2_biases.elements.size(); i++) {
        output << h2_biases.elements[i] << " ";
    }

    output << "\n";

    for (int i = 0; i < out_biases.elements.size(); i++) {
        output << out_biases.elements[i] << " ";
    }

    output.close();

    return true;
}