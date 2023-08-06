#ifndef INPUT_LAYER_HPP
#define INPUT_LAYER_HPP

#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas.h>

#include "../include/memory_manager.hpp"

using namespace std;

class InputLayer {
public:
    InputLayer();
    InputLayer(cudnnHandle_t* cudnnHandle, string modelPath, DNNLayer* dnnLayer);
    ~InputLayer();

    void trainForward(vector<unsigned int> tokenIds);

    cudnnTensorDescriptor_t getOutputTensorDescriptor();
    DNNLayer::Block* getOutputBlock();

private:
    cudnnHandle_t* cudnnHandle;
    DNNLayer* dnnLayer;

    cudnnTensorDescriptor_t outputTensorDescriptor;
    DNNLayer::Block outputBlock;

    string modelPath;
};

#endif