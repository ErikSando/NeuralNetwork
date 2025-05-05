#pragma once

#include <iostream>
#include <array>
#include <vector>

#include <Globals.h>

struct ImageData {
    int digit;
    std::array<float, N_INPUT_NODES> pixels;
};

namespace DataParser {
    ImageData* ParseInputFile(const int digit, const std::string& filepath); // for processing a single data file
    ImageData* GetRowImageData(const int row, const std::string& csv_path);
    
    void PrintData(ImageData* data);
}