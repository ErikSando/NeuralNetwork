#include <chrono>
#include <iostream>
#include <sstream>

#include "Globals.h"
#include "DataParser.h"
#include "NeuralNetwork.h"
#include "Utility.h"

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
        // TODO: make the commands more flexible with which training data is used
        // Also make a list of networks which the user can add to, to train for different data sets
        // And finally fix the uplo
        else if (cmd == "help") {
            std::cout << "help:\n- Shows this menu." << std::endl;
            std::cout << "train [no. iterations]:\n- Train the model with the number of iterations specified. One iteration uses one image." << std::endl;
            //std::cout << "test [mo. iterations]:\n- Test the model with the number of iterations specified. One iteration uses one image. The accuracy will be printed at the end." << std::endl;
            //std::cout << "bs [batch size]:\n- Set the batch size of the model. Default value is 32. Small values will significantly limit training speed." << std::endl;
            std::cout << "lr [learning rate]:\n- Set the learning rate of the model. Default value is 0.001. Large values may break the model." << std::endl;
            std::cout << "save [save path]\n- Save model data to the specified save file." << std::endl;
            std::cout << "load [save path]:\n- (currently not working correctly) Load model data from the specified save file." << std::endl;
            //std::cout << "id [image data path] [optional: correct digit]\n- Identify a digit in the data provided. If the correct digit is provided, a loss value will be printed." << std::endl;
            std::cout << "read [no. lines] [optional: path]\n- Test the time taken to read the specified number of lines from a file, and convert into input data. If no path is given, the default training CSV will be used." << std::endl;
        }
        else if (cmd == "save") {
            if (args.size() < 2) {
                std::cout << "Insufficient arguments" << std::endl;
                std::cout << "Usage: save [save path]" << std::endl;
                continue;
            }

            std::string savepath = args.at(1);

            if (network.SaveModel(savepath)) {
                std::cout << "Saved model data to " << savepath << std::endl;
            }
        }
        else if (cmd == "load") {
            if (args.size() < 2) {
                std::cout << "Insufficient arguments" << std::endl;
                std::cout << "Usage: upload [save path]" << std::endl;
                continue;
            }

            std::string savepath = args.at(1);

            network.LoadModel(savepath);

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
            
            std::array<uint8_t, N_INPUT_NODES * BATCH_SIZE> inputs;
            const ImageData* data = DataParser::ParseInputFile(digit, datapath);

            for (int i = 0; i < BATCH_SIZE; i++) {
                std::copy(data->pixels.begin(), data->pixels.end(), inputs.begin() + i * BATCH_SIZE * sizeof(uint8_t));
            }

            delete data;

            std::array<float, N_OUTPUT_NODES * BATCH_SIZE> _outputs = network.GetOutputs(inputs);
            std::array<float, N_OUTPUT_NODES> outputs; // first 10

            for (int i = 0; i < N_OUTPUT_NODES; i++) {
                outputs[i] = _outputs[i];
            }

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
                std::array<uint8_t, N_OUTPUT_NODES> true_outputs = Utility::GetTrueOutputs(digit);

                #ifndef NDEBUG

                for (int i = 0; i < 10; i++) {
                    uint8_t expected = (i == data->digit) ? 1 : 0;
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
        // else if (cmd == "bs") {
        //     if (args.size() < 2) {
        //         std::cout << "Insufficient arguments" << std::endl;
        //         std::cout << "Usage: bs [batch size]" << std::endl;
        //         continue;
        //     }

        //     float batch_size = std::stoi(args.at(1));

        //     network.SetBatchSize(batch_size);
        //     std::cout << "Set batch size to " << batch_size << std::endl;
        // }
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
        
            delete t_data;
        }
        else if (cmd == "read") {
            if (args.size() < 2) {
                std::cout << "Insufficient arguments" << std::endl;
                std::cout << "Usage: read [no. lines] [optional: path]" << std::endl;
                continue;
            }

            int lines = std::stoi(args.at(1));
            std::string path = "data/mnistdata/mnist_train.csv";

            if (args.size() > 2) {
                path = args.at(3);
            }

            auto start = std::chrono::high_resolution_clock::now();

            for (int l = 1; l < lines; l++) {
                const ImageData* data = DataParser::GetRowData((l - 1) % TRAINING_ROWS + 1, path);
                delete data;
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
            std::cout << "Processed " << lines << " lines in " << duration << " ms." << std::endl;
        }
        else if (cmd == "read-batch") {
            if (args.size() < 2) {
                std::cout << "Insufficient arguments" << std::endl;
                std::cout << "Usage: read-batch [no. batches] [optional: path]" << std::endl;
                continue;
            }

            int batches = std::stoi(args.at(1));
            std::string path = "data/mnistdata/mnist_train.csv";

            if (args.size() > 2) {
                path = args.at(3);
            }

            auto start = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < batches; b++) {
                //std::cout << (b * BATCH_SIZE + 1) << std::endl;
                std::array<ImageData*, BATCH_SIZE> data = DataParser::GetBatchedTrainingData(b * BATCH_SIZE + 1, path);
                
                for (int i = 0; i < BATCH_SIZE; i++) {
                    delete data[i];
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
            std::cout << "Processed " << batches << " batches in " << duration << " ms." << std::endl;
        }
    }
}