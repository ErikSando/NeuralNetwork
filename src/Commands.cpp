#include <iostream>
#include <sstream>

#include <Globals.h>
#include <NeuralNetwork.h>
#include <DataParser.h>
#include <Utility.h>

void StartConsoleLoop() {
    NeuralNetwork network;

    std::string command;

    while (command != "exit") {
        std::cout << "> ";
        std::getline(std::cin, command);

        std::istringstream iss(command);
        std::vector<std::string> args;
        std::string arg;

        while (iss >> arg) {
            args.emplace_back(arg);
        }

        if (args.size() < 1) continue;

        std::string cmd = args.at(0);

        if (cmd == "exit") {
            break;
        }
        else if (cmd == "save") {
            if (args.size() < 2) {
                std::cout << "Insufficient arguments" << std::endl;
                std::cout << "Usage: save [save path]" << std::endl;
                continue;
            }

            std::string savepath = args.at(1);

            network.SaveModel(savepath);
        }
        else if (cmd == "upload") {
            if (args.size() < 2) {
                std::cout << "Insufficient arguments" << std::endl;
                std::cout << "Usage: upload [save path]" << std::endl;
                continue;
            }

            std::string savepath = args.at(1);

            network.UploadModel(savepath);
        }
        else if (cmd == "id") {
            if (args.size() < 2) {
                std::cout << "Insufficient arguments" << std::endl;
                std::cout << "Usage: id [image data path] [optional: digit]" << std::endl;
                continue;
            }

            std::string datapath = args.at(1);

            bool digit_specified = false;
            int digit = 0;
            if (args.size() >= 3) {
                digit_specified = true;
                digit = std::stoi(args.at(2));
            }
            
            ImageData* data = DataParser::ParseInputFile(digit, datapath);

            std::array<float, N_OUTPUT_NODES> outputs = network.GetOutputs(data->pixels);

            float largest = 0;
            int l_digit = -1;

            for (int i = 0; i < N_OUTPUT_NODES; i++) {
                std::cout << i << ": " << outputs[i] << std::endl;

                if (outputs[i] > largest) {
                    largest = outputs[i];
                    l_digit = i;
                }
            }

            std::cout << "Identified digit: " << l_digit << " (" << largest << " probability)" << std::endl;

            if (digit_specified) {
                std::array<float, N_OUTPUT_NODES> true_outputs = Utility::GetTrueOutputs(digit);
                float loss = Utility::LossFunctions::CategoricalCrossEntropy(true_outputs, outputs);
                std::cout << "Loss: " << loss << " (lower means more accurate)" << std::endl;
            }
        }
    }
}