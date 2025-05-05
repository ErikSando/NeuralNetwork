#pragma once

#include <cassert>

#define IMAGE_ROWS 28
#define IMAGE_COLUMNS 28

#define N_INPUT_NODES (IMAGE_ROWS * IMAGE_COLUMNS)
#define N_OUTPUT_NODES 10

#define N_HIDDEN_LAYERS 2
#define N_NODES_PER_HIDDEN_LAYER 50

#define N_HIDDEN_NODES (N_HIDDEN_LAYERS * N_NODES_PER_HIDDEN_LAYER)

#define N_LAYERS (N_HIDDEN_LAYERS + 1) // excludes input layer

#define TRAINING_ROWS 60000

extern void StartConsoleLoop();