#include <cudnn_backend.h>
#include <cudnn_ops_infer.h>
#include <filesystem>
#include <string>

#include "../include/cuda_utils.hpp"
#include "../include/bert.hpp"
#include "../include/input_layer.hpp"
#include "../include/memory_manager.hpp"
#include "../include/profiler.hpp"
#include "../include/weight_loader.hpp"

using namespace std;

InputLayer::InputLayer() {

}

InputLayer::InputLayer(cudnnHandle_t* cudnnHandle, string modelPath, DNNLayer* dnnLayer)
    : cudnnHandle(cudnnHandle),
    modelPath(modelPath),
    dnnLayer(dnnLayer) {

    Profiler::push("cudaSetup");

    this->outputTensorDescriptor = cudnnTensorDescriptor_t();
    checkCUDNN(cudnnCreateTensorDescriptor(&this->outputTensorDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->outputTensorDescriptor,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BERT::BATCH_SIZE, 1,
        BERT::MAX_INPUT_LENGTH, BERT::HIDDEN_SIZE));

    size_t floatsInEmbeddings = BERT::BATCH_SIZE * 1 * BERT::MAX_INPUT_LENGTH * BERT::HIDDEN_SIZE;
    size_t bytesInEmbeddings = floatsInEmbeddings * sizeof(float);

    this->outputBlock = dnnLayer->allocateBlock(bytesInEmbeddings);
    this->outputBlock.loadRandom();

    Profiler::pop();
}

InputLayer::~InputLayer() {
    
}

void InputLayer::trainForward(vector<unsigned int> tokenIds) {

    dnnLayer->unspill();
    
    // No computation is done by this layer, it simply holds a random input
    // tensor

}

cudnnTensorDescriptor_t InputLayer::getOutputTensorDescriptor() {
    return this->outputTensorDescriptor;
}

DNNLayer::Block* InputLayer::getOutputBlock() {
    return &this->outputBlock;
}