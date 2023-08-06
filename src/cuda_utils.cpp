#include "../include/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_ops_infer.h>

// Error handling
// Adapted from https://github.com/tbennun/cudnn-training

void checkCUDNN(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
      spdlog::error("CUDNN failure! Status code:", status);
      spdlog::error(cudnnGetErrorString(status));
      exit(1);
    }
}

void checkCudaErrors(cudaError status) {
    if (status != 0) {
      spdlog::error("CUDA failure! Status code: {}", status);
      spdlog::error(cudaGetErrorString(status));
      exit(1);
    }
}