#pragma once

// Standard libs
#include <cstdlib>
#include <iosfwd>
// Cuda
#include "cuda_runtime.h"
#include "libs/helper_cuda.h"
#include "loggers.h"

// Function to check cuda requirements
bool initCuda();