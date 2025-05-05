#include <cmath>

#include <DataParser.h>
#include <Utility.h>

ImageData* DataParser::ParseInputFile(const int digit, const std::string& filepath) {
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
            data->pixels[pixel_index] = std::stof(current_val);
            current_val = "";
            pixel_index++;
        }
    }

    return data;
}

ImageData* DataParser::GetRowImageData(const int row, const std::string& csv_path) {
    ImageData* imagedata = new ImageData;

    std::string raw_text = Utility::ReadLine(row, csv_path);
    std::string current_val = "";

    int column = 0;

    for (int i = 0; i < raw_text.length();) {
        char c = raw_text.at(i++);

        while (i < raw_text.length() && (std::isdigit(c) || c == '.')) {
            current_val += c;
            c = raw_text.at(i++);
        }

        if (current_val != "") {
            if (column) imagedata->pixels[column] = std::stoi(current_val);
            else imagedata->digit = std::stoi(current_val);

            current_val = "";
            column++;
        }
    }

    return imagedata;
}

// doesnt work for negative numbers, if u want to use it for negative numbers use std::floor(std::log10(n)) + 1
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