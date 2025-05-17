#include <fstream>
#include <iostream>

#include "NeuralNetwork.h"
#include "Utility.h"

void NeuralNetwork::LoadModel(const std::string& savepath) {
    //std::string raw_text = Utility::ReadFile(savepath);

    int tl = N_HIDDEN_LAYERS * 2 + 2; // total layers excluding input layer

    Matrix* matrices[tl] = { &h1_weights, &h2_weights, &out_weights, &h1_biases, &h2_biases, &out_biases };
    std::string types[tl] = { "weights", "weights", "weights", "biases", "biases", "biases" };
    std::string layer_name[tl / 2] = { "hidden layer one", "hidden layer two", "the output layer" };
    std::string idk_what_to_call_this[tl / 2] = { "randomised", "randomised", "as zero" };

    //int i = 0;

    //char c = raw_text.at(i++);

    for (int m = 0; m < tl; m++) {
        std::string raw_text = Utility::ReadLine(m + 1, savepath);
        int i = 0;

        std::string current_val = "";
        int m_i = 0;
        Matrix* mat = matrices[m];

        char c = raw_text.at(i++);

        while (c != '\n') {
            while (std::isdigit(c) || c == '-' || c == '.') {
                current_val += c;
                c = raw_text.at(i++);
            }

            if (current_val != "") {
                mat->elements[m_i++] = std::stof(current_val);
                current_val = "";

                if (m_i >= mat->elements.size()) {
                    // while(c != '\n') { // skip to next line
                    //     c = raw_text.at(i++);
                    // }

                    break;
                }
            }

            c = raw_text.at(i++);
        }

        if (m_i < mat->elements.size()) {
            std::cout << "Not enough " << types[m] << " for " << layer_name[m % (tl / 2)] << " (" << m_i << "/" << mat->elements.size() << "), some " << types[m] << " remain " << idk_what_to_call_this[m % (tl / 2)] << "." << std::endl;
        }
    }

    // std::string current_val = "";
    // int h1_wi = 0;

    // while (c != '\n') {
    //     while (std::isdigit(c) || c == '-' || c == '.') {
    //         current_val += c;
    //         c = raw_text.at(i++);
    //     }

    //     if (current_val != "") {
    //         h1_weights.elements[h1_wi++] = std::stof(current_val);
    //         current_val = "";

    //         if (h1_wi >= h1_weights.elements.size()) {
    //             // while(c != '\n') { // skip to next line
    //             //     c = raw_text.at(i++);
    //             // }

    //             break;
    //         }
    //     }

    //     c = raw_text.at(i++);
    // }

    // if (h1_wi < h1_weights.elements.size()) {
    //     std::cout << "Not enough weights for hidden layer one (" << h1_wi << "/" << h1_weights.elements.size() << "), some weights remain randomised." << std::endl;
    // }

    // current_val = "";
    // int h2_wi = 0;

    // while (c != '\n') {
    //     while (std::isdigit(c) || c == '-' || c == '.') {
    //         current_val += c;
    //         c = raw_text.at(i++);
    //     }

    //     if (current_val != "") {
    //         std::cout << h2_wi << std::endl;
    //         std::cout << current_val << std::endl;
    //         h2_weights.elements[h2_wi++] = std::stof(current_val);
    //         current_val = "";

    //         if (h2_wi >= h2_weights.elements.size()) {
    //             // while(c != '\n') { // skip to next line
    //             //     c = raw_text.at(i++);
    //             // }

    //             break;
    //         }
    //     }

    //     c = raw_text.at(i++);
    // }

    // if (h2_wi < h2_weights.elements.size()) {
    //     std::cout << "Not enough weights for hidden layer two (" << h2_wi << "/" << h2_weights.elements.size() << "), some weights remain randomised." << std::endl;
    // }

    // current_val = "";
    // int out_wi = 0;

    // while (c != '\n') {
    //     while (std::isdigit(c) || c == '-' || c == '.') {
    //         current_val += c;
    //         c = raw_text.at(i++);
    //     }

    //     if (current_val != "") {
    //         out_weights.elements[out_wi++] = std::stof(current_val);
    //         current_val = "";

    //         if (out_wi >= out_weights.elements.size()) {
    //             // while(c != '\n') { // skip to next line
    //             //     c = raw_text.at(i++);
    //             // }

    //             break;
    //         }
    //     }

    //     c = raw_text.at(i++);
    // }

    // if (out_wi < out_weights.elements.size()) {
    //     std::cout << "Not enough weights for the output layer (" << out_wi << "/" << out_weights.elements.size() << "), some weights remain randomised." << std::endl;
    // }

    // current_val = "";
    // int h1_bi = 0;

    // while (c != '\n') {
    //     while (std::isdigit(c) || c == '-' || c == '.') {
    //         current_val += c;
    //         c = raw_text.at(i++);
    //     }

    //     if (current_val != "") {
    //         h1_biases.elements[h1_bi++] = std::stof(current_val);
    //         current_val = "";

    //         if (h1_bi >= h1_biases.elements.size()) {
    //             // while(c != '\n') { // skip to next line
    //             //     c = raw_text.at(i++);
    //             // }

    //             break;
    //         }
    //     }

    //     c = raw_text.at(i++);
    // }

    // if (h1_bi < h1_biases.elements.size()) {
    //     std::cout << "Not enough biases for hidden layer one (" << h1_bi << "/" << h1_biases.elements.size() << "), some biases remain as zero." << std::endl;
    // }

    // current_val = "";
    // int h2_bi = 0;

    // while (c != '\n') {
    //     while (std::isdigit(c) || c == '-' || c == '.') {
    //         current_val += c;
    //         c = raw_text.at(i++);
    //     }

    //     if (current_val != "") {
    //         h2_biases.elements[h2_bi++] = std::stof(current_val);
    //         current_val = "";

    //         if (h2_bi >= h2_biases.elements.size()) {
    //             // while(c != '\n') { // skip to next line
    //             //     c = raw_text.at(i++);
    //             // }

    //             break;
    //         }
    //     }

    //     c = raw_text.at(i++);
    // }

    // if (h2_bi < h2_biases.elements.size()) {
    //     std::cout << "Not enough biases for hidden layer two (" << h2_bi << "/" << h2_biases.elements.size() << "), some biases remain as zero." << std::endl;
    // }

    // current_val = "";
    // int out_bi = 0;

    // while (c != '\n') {
    //     while (std::isdigit(c) || c == '-' || c == '.') {
    //         std::cout << "Adding " << c << " to value" << std::endl;
    //         current_val += c;
    //         c = raw_text.at(i++);
    //     }

    //     if (current_val != "") {
    //         std::cout << current_val << std::endl;
    //         out_biases.elements[out_bi++] = std::stof(current_val);
    //         current_val = "";

    //         if (out_bi >= out_biases.elements.size()) {
    //             // while(c != '\n') { // skip to next line
    //             //     c = raw_text.at(i++);
    //             // }
                
    //             break;
    //         }
    //     }

    //     c = raw_text.at(i++);
    // }

    // if (out_bi < out_biases.elements.size()) {
    //     std::cout << "Not enough biases for the output layer (" << out_bi << "/" << out_biases.elements.size() << "), some biases remain as zero." << std::endl;
    // }
}