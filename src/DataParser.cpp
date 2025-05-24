#include <cmath>
#include <cstdint>

#include "DataParser.h"
#include "Utility.h"

const ImageData* DataParser::ParseInputFile(const int digit, const std::string& filepath) {
    ImageData* data = new ImageData{ .digit = digit };

    std::string raw_text = Utility::ReadFile(filepath);
    std::string current_val = "";

    int pixel_index = 0;

    for (int i = 0; i < raw_text.length();) {
        char c = raw_text.at(i++);

        while (i < raw_text.length() && (std::isdigit(c) || c == '.')) {
            current_val += c;
            c = raw_text.at(i++);
        }

        if (current_val != "") {
            data->pixels[pixel_index] = static_cast<uint8_t>(std::stoi(current_val));
            current_val = "";
            pixel_index++;
        }
    }

    return data;
}

const std::array<ImageData*, BATCH_SIZE> DataParser::GetBatchedTrainingData(const int start_row, const std::string& csv_path) {
    std::array<ImageData*, BATCH_SIZE> image_data;

    for (int i = 0; i < BATCH_SIZE; i++) {
        image_data[i] = new ImageData;

        std::string raw_text = Utility::ReadLine((start_row + i - 1) % TRAINING_ROWS + 1, csv_path);
        std::string current_val = "";

        int column = 0;

        for (int j = 0; j < raw_text.length();) {
            char c = raw_text.at(j++);

            while (j < raw_text.length() && (std::isdigit(c)/* || c == '.'*/)) {
                current_val += c;
                c = raw_text.at(j++);
            }

            if (current_val != "") {
                if (column) image_data[i]->pixels[column] = static_cast<uint8_t>(std::stoi(current_val));
                else image_data[i]->digit = std::stoi(current_val);

                current_val = "";
                column++;
            }
        }
    }

    return image_data;
}

const ImageData* DataParser::GetRowData(const int row, const std::string& csv_path) {
    ImageData* image_data = new ImageData;
        
    std::string raw_text = Utility::ReadLine(row, csv_path);
    std::string current_val = "";

    int column = 0;

    for (int j = 0; j < raw_text.length();) {
        char c = raw_text.at(j++);

        while (j < raw_text.length() && (std::isdigit(c)/* || c == '.'*/)) {
            current_val += c;
            c = raw_text.at(j++);
        }

        if (current_val != "") {
            if (column) image_data->pixels[column] = static_cast<uint8_t>(std::stoi(current_val));
            else image_data->digit = std::stoi(current_val);

            current_val = "";
            column++;
        }
    }

    return image_data;
}

// doesnt work for negative numbers, if u want to use it for negative numbers use std::floor(std::log10(std::abs(n))) + 1
int num_digits(int n) {
    if (n == 0) return 1;

    return std::floor(std::log10(n)) + 1;
}

void DataParser::PrintData(ImageData* data) {
    std::cout << "Digit: " << data->digit << std::endl;

    for (int r = 0; r < IMAGE_ROWS; r++) {
        for (int c = 0; c < IMAGE_COLUMNS; c++) {
            int pixel = data->pixels[r * IMAGE_COLUMNS + c];
            int n_digits = num_digits(pixel);

            for (int i = 0; i < 4 - n_digits; i++) {
                std::cout << " ";
            }

            std::cout << pixel;
        }

        std::cout << std::endl;
    }
}