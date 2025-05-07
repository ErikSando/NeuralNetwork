#pragma once

#include <cassert>

#define IMAGE_ROWS 28
#define IMAGE_COLUMNS 28

#define N_INPUT_NODES (IMAGE_ROWS * IMAGE_COLUMNS)
#define N_OUTPUT_NODES 10

#define N_HIDDEN_LAYERS 2
#define HIDDEN_LAYER_1_SIZE 128
#define HIDDEN_LAYER_2_SIZE (HIDDEN_LAYER_1_SIZE / 2)

#define N_HIDDEN_NODES (HIDDEN_LAYER_1_SIZE + HIDDEN_LAYER_2_SIZE)

#define N_LAYERS (N_HIDDEN_LAYERS + 1) // excludes input layer

#define TRAINING_ROWS 60000
#define TESTING_ROWS 10000

extern void StartConsoleLoop();