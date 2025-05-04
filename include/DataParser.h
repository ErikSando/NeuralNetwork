#pragma once

#include <iostream>
#include <vector>
#include <array>

#include <Globals.h>

struct ImageData {
    int number;
    std::array<float, N_INPUT_NODES> pixels;
};

namespace DataParser {
    ImageData* ParseInputFile(const int number, const std::string& filepath); // for processing a single data file
    ImageData* GetRowImageData(const int row, const std::string& csv_path);
    
    void PrintData(ImageData* data);
}