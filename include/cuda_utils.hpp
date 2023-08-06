#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas.h>

#include "spdlog/spdlog.h"

void checkCUDNN(cudnnStatus_t status);
void checkCudaErrors(cudaError status);

#endif