#include <algorithm>

#include <DataParser.h>
#include <NeuralNetwork.h>
#include <Utility.h>

void NeuralNetwork::SetLearningRate(const float lr) {
    learning_rate = lr;
}

void NeuralNetwork::SetBatchSize(const int bs) {
    batch_size = bs;
}

void NeuralNetwork::Train(const int iterations, const std::string& trainpath, const bool newpath) {
    if (newpath) current_row = 1;

    for (int i = 0; i < iterations; i++) {
        ImageData* data = DataParser::GetRowImageData(current_row, trainpath);
    }
}