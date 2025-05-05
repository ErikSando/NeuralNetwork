#include <iostream>
#include <cmath>
#include <chrono>

#include <Globals.h>
#include <DataParser.h>
#include <NeuralNetwork.h>
#include <Utility.h>

int main() {
    Random::Init();

    StartConsoleLoop();

    // no training process has been implemented yet
    // NeuralNetwork network;

    // std::string path = "./data/mnistdata/mnist_test.csv";\
    // //std::string custom_inp_path = "./data/three";

    // //ImageData* test_data = DataParser::ParseInputFile(3, custom_inp_path);

    // //DataParser::PrintData(test_data);

    // ImageData* data = DataParser::GetRowImageData(1, path);
    // ImageData* data2 = DataParser::GetRowImageData(5, path);

    // DataParser::PrintData(data);
    // DataParser::PrintData(data2);

    // // speed test
    // // int n_rows = 5000;
    // // std::cout << "Beginning speed test..." << std::endl;

    // // auto start = std::chrono::high_resolution_clock::now();

    // // for (int i = 1; i <= n_rows; i++) {
    // //     ImageData* _data = DataParser::GetRowImageData(i, path);
    // //     delete _data;
    // // }

    // // auto end = std::chrono::high_resolution_clock::now();
    // // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // // std::cout << "Processed " << n_rows << " rows of data in " << duration << " ms" << std::endl;

    // // int size_n = sizeof(Node);
    // // int size_c = sizeof(Connection);
    // // int size_l = sizeof(Layer);

    // // int t_size_n = size_n * N_INPUT_NODES + size_n * hidden_layers * noders_per_hidden_layer + size_n * N_OUTPUT_NODES;
    // // int t_size_c = size_c * N_INPUT_NODES * noders_per_hidden_layer + size_c * noders_per_hidden_layer * noders_per_hidden_layer * (hidden_layers - 1) + size_c * noders_per_hidden_layer * N_OUTPUT_NODES;
    // // int t_size_l = size_l * (hidden_layers + 2);

    // // std::cout << "Calculated size: " << (t_size_n + t_size_c + t_size_l) << " bytes" << std::endl;
    // std::cout << "Neural Network size: " << sizeof(network) << " bytes" << std::endl;

    // std::array<float, N_OUTPUT_NODES> output1 = network.GetOutputs(data->pixels);
    // std::array<float, N_OUTPUT_NODES> output2 = network.GetOutputs(data2->pixels);
    // // std::array<float, N_OUTPUT_NODES> output3 = network.GetOutputs(data2->pixels);
    // // std::array<float, N_OUTPUT_NODES> output4 = network.GetOutputs(data->pixels);

    // std::cout << "Results for test 1" << std::endl;

    // for (int i = 0; i < N_OUTPUT_NODES; i++) {
    //     std::cout << i << ": " << output1[i] << std::endl;
    // }

    // std::cout << "Results for test 2" << std::endl;

    // for (int i = 0; i < N_OUTPUT_NODES; i++) {
    //     std::cout << i << ": " << output2[i] << std::endl;
    // }

    // // for (int i = 0; i < N_OUTPUT_NODES; i++) {
    // //     std::cout << i << ": " << output3[i] << ", " << confidences3[i] << std::endl;
    // // }

    // // for (int i = 0; i < N_OUTPUT_NODES; i++) {
    // //     std::cout << i << ": " << output4[i] << ", " << confidences4[i] << std::endl;
    // // }

    // std::array<float, N_OUTPUT_NODES> true_output1 = Utility::GetTrueOutputs(data->digit);
    // std::array<float, N_OUTPUT_NODES> true_output2 = Utility::GetTrueOutputs(data2->digit);

    // float loss1 = Utility::LossFunctions::CategoricalCrossEntropy(true_output1, output1);
    // float loss2 = Utility::LossFunctions::CategoricalCrossEntropy(true_output2, output2);

    // std::cout << "Loss for data 1: " << loss1 << std::endl;
    // std::cout << "Loss for data 2: " << loss2 << std::endl;

    //delete test_data;
    // delete data;
    // delete data2;

    return 0;
}