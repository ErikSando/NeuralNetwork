#include <iostream>
#include <fstream>

#include <NeuralNetwork.h>
#include <Utility.h>

bool NeuralNetwork::SaveModel(const std::string& savepath) {
    std::ofstream output(savepath, std::ios::trunc);

    output.close();

    return false;
}