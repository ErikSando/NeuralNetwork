#include <iostream>
#include <sstream>

#include <Globals.h>
#include <DataParser.h>
#include <NeuralNetwork.h>
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
        else if (cmd == "help") {

        }
        else if (cmd == "save") {
            if (args.size() < 2) {
                std::cout << "Insufficient arguments" << std::endl;
                std::cout << "Usage: save [save path]" << std::endl;
                continue;
            }

            std::string savepath = args.at(1);

            network.SaveModel(savepath);

            std::cout << "Saved model data to " << savepath << std::endl;
        }
        else if (cmd == "upload") {
            if (args.size() < 2) {
                std::cout << "Insufficient arguments" << std::endl;
                std::cout << "Usage: upload [save path]" << std::endl;
                continue;
            }

            std::string savepath = args.at(1);

            network.UploadModel(savepath);

            std::cout << "Uploaded data from " << savepath << std::endl;
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

            // repeated code, TODO: prevent repeating
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

                #ifndef NDEBUG

                for (int i = 0; i < 10; i++) {
                    float expected = (i == data->digit) ? 1.0f : 0.0f;
                    assert(true_outputs[i] == expected);
                }

                #endif

                float loss = Utility::LossFunctions::CategoricalCrossEntropy(true_outputs, outputs);
                std::cout << "CCE Loss: " << loss << " (lower means more accurate)" << std::endl;
            }
        }
        else if (cmd == "lr") {
            if (args.size() < 2) {
                std::cout << "Insufficient arguments" << std::endl;
                std::cout << "Usage: lr [learning rate]" << std::endl;
                continue;
            }

            int learning_rate = std::stoi(args.at(1));

            network.SetLearningRate(learning_rate);
            std::cout << "Set learning rate to " << learning_rate << std::endl;
        }
        else if (cmd == "train") {
            int iterations = 1;
            
            if (args.size() > 1) {
                iterations = std::stoi(args.at(1));
            }
    
            std::cout << "Training..." << std::endl;
    
            network.Train(iterations, "./data/mnistdata/mnist_train.csv");
    
            std::cout << "Training complete for " << iterations << " iterations." << std::endl;
        }
        else if (cmd == "test") {
            int iterations = 1;
            
            if (args.size() > 1) {
                iterations = std::stoi(args.at(1));
            }
    
            std::cout << "Testing..." << std::endl;
    
            TestingData* t_data = network.Test(iterations, "./data/mnistdata/mnist_test.csv");
    
            std::cout << "Testing complete for " << iterations << " iterations." << std::endl;

            int c = t_data->correct;
            int ic = t_data->incorrect;

            std::cout << "Correct: " << c << ", incorrect: " << ic << std::endl;
            std::cout << "Accuracy: " << (100 * ((float)c / (float)(c + ic))) << "% (" << c << "/" << (c + ic) << ")" << std::endl;
        }
    }
}