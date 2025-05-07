#include <iostream>
#include <sstream>
#include <chrono>

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
            std::cout << "help:\n- Shows this menu." << std::endl;
            std::cout << "train [iterations]:\n- Train the model with the number of iterations specified. One iteration uses one image." << std::endl;
            std::cout << "test [iterations]:\n- Test the model with the number of iterations specified. One iteration uses one image. The accuracy will be printed at the end." << std::endl;
            std::cout << "lr [learning rate]:\n- Set the learning rate of the model. Default value is 0.001. Large values may break the model." << std::endl;
            std::cout << "save [save path]\n- Save model data to the specified save file." << std::endl;
            std::cout << "upload [save path]:\n- (currently not working correctly) Upload model data from the specified save file." << std::endl;
            std::cout << "id [image data path] [optional: correct digit]\n- Identify a digit in the data provided. If the correct digit is provided, a loss value will be printed." << std::endl;
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

                float loss = Utility::Loss::CategoricalCrossEntropy(true_outputs, outputs);
                std::cout << "CCE Loss: " << loss << " (lower means more accurate)" << std::endl;
            }
        }
        else if (cmd == "lr") {
            if (args.size() < 2) {
                std::cout << "Insufficient arguments" << std::endl;
                std::cout << "Usage: lr [learning rate]" << std::endl;
                continue;
            }

            float learning_rate = std::stof(args.at(1));

            network.SetLearningRate(learning_rate);
            std::cout << "Set learning rate to " << learning_rate << std::endl;
        }
        else if (cmd == "train") {
            int iterations = 1;
            
            if (args.size() > 1) {
                iterations = std::stoi(args.at(1));
            }
    
            std::cout << "Training..." << std::endl;

            auto start = std::chrono::high_resolution_clock::now();
    
            network.Train(iterations, "./data/mnistdata/mnist_train.csv");
    
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "Training complete for " << iterations << " iterations in " << duration << " ms." << std::endl;
        }
        else if (cmd == "test") {
            int iterations = 1;
            
            if (args.size() > 1) {
                iterations = std::stoi(args.at(1));
            }
    
            std::cout << "Testing..." << std::endl;

            auto start = std::chrono::high_resolution_clock::now();
    
            TestingData* t_data = network.Test(iterations, "./data/mnistdata/mnist_test.csv");

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
            std::cout << "Testing complete for " << iterations << " iterations in " << duration << " ms." << std::endl;

            int c = t_data->correct;
            int ic = t_data->incorrect;

            std::cout << "Correct: " << c << ", incorrect: " << ic << std::endl;
            std::cout << "Accuracy: " << (100 * ((float)c / (float)(c + ic))) << "% (" << c << "/" << (c + ic) << ")" << std::endl;
        }
    }
}