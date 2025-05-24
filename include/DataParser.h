#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

#include "Globals.h"

struct ImageData {
    int digit;
    std::array<uint8_t, N_INPUT_NODES> pixels;
};

namespace DataParser {
    const ImageData* ParseInputFile(const int digit, const std::string& filepath); // for processing a single data file
    const std::array<ImageData*, BATCH_SIZE> GetBatchedTrainingData(const int start_row, const std::string& csv_path);
    const ImageData* GetRowData(const int row, const std::string& csv_path);
    //const std::vector<ImageData*> GetRowsImageData(const int row, const int n_rows, const std::string& csv_path);
    
    void PrintData(ImageData* data);
}