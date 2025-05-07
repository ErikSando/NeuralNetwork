#include <iostream>
#include <fstream>

#include <NeuralNetwork.h>
#include <Utility.h>

bool NeuralNetwork::SaveModel(const std::string& savepath) {
    std::ofstream output(savepath, std::ios::trunc);

    
}

void NeuralNetwork::UploadModel(const std::string& savepath) {
    std::string raw_text = Utility::ReadFile(savepath);

    
}