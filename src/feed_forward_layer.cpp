#include "../include/feed_forward_layer.hpp"

#include <string>

#include <cudnn.h>

#include "../include/bert.hpp"
#include "../include/cuda_utils.hpp"
#include "../include/memory_manager.hpp"
#include "../include/profiler.hpp"
#include "../include/weight_loader.hpp"

FeedForwardLayer::FeedForwardLayer(cudnnHandle_t* cudnnHandle, DNNLayer* dnnLayer)
    : cudnnHandle(cudnnHandle),
    dnnLayer(dnnLayer) {

    Profiler::push("cudaSetup");

    // Do the feed-forward operation using convolutions
    this->ffnConvolutionDescriptor = cudnnConvolutionDescriptor_t();
    checkCUDNN(cudnnCreateConvolutionDescriptor(&this->ffnConvolutionDescriptor));

    checkCUDNN(cudnnSetConvolution2dDescriptor(
        this->ffnConvolutionDescriptor,
        0, // pad_h
        0, // pad_w
        1, // vertical filter stride
        1, // horizontal filter stride
        1, // filter height dilation (1 = no dilation)
        1, // filter width dilation
        CUDNN_CONVOLUTION, // Convolution mode
        CUDNN_DATA_FLOAT));
    
    // Set up input, filter, and output tensors
    this->xDesc                  = cudnnTensorDescriptor_t();
    this->wDesc                  = cudnnFilterDescriptor_t();
    this->outputTensorDescriptor = cudnnTensorDescriptor_t();
    checkCUDNN(cudnnCreateTensorDescriptor(&this->xDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&this->wDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&this->outputTensorDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->xDesc,                  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BERT::BATCH_SIZE, 1, BERT::HIDDEN_SIZE, BERT::MAX_INPUT_LENGTH));
    checkCUDNN(cudnnSetFilter4dDescriptor(this->wDesc,                  CUDNN_DATA_FLOAT,  CUDNN_TENSOR_NCHW,               1, 1,                 1,                      1));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->outputTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BERT::BATCH_SIZE, 1, BERT::HIDDEN_SIZE, BERT::MAX_INPUT_LENGTH));

    this->convolutionInput       = dnnLayer->allocateBlock(BERT::BATCH_SIZE * 1 * BERT::HIDDEN_SIZE * BERT::MAX_INPUT_LENGTH * sizeof(float));
    this->convolutionFilter      = dnnLayer->allocateBlock(BERT::BATCH_SIZE * 1 * BERT::HIDDEN_SIZE * BERT::MAX_INPUT_LENGTH * sizeof(float));
    this->outputTensorDeviceData = dnnLayer->allocateBlock(BERT::BATCH_SIZE * 1 * BERT::HIDDEN_SIZE * BERT::MAX_INPUT_LENGTH * sizeof(float));
    this->convolutionInput.loadRandom();
    this->convolutionFilter.loadRandom();
    
    // Set up data for the convolution and execute it
    // Initialize input and output scaling tensors
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        *this->cudnnHandle,
        this->xDesc,
        this->wDesc,
        this->ffnConvolutionDescriptor,
        this->outputTensorDescriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        &this->forwardWorkspaceSize));

    this->forwardWorkspace = dnnLayer->allocateBlock(forwardWorkspaceSize);
    this->outputTensorDeviceData = dnnLayer->allocateBlock(BERT::BATCH_SIZE * 1 * BERT::HIDDEN_SIZE * BERT::MAX_INPUT_LENGTH * sizeof(float));

    // Allocate space for the fully connected layer's gradients
    this->dwDesc = cudnnFilterDescriptor_t();
    cudnnCreateFilterDescriptor(&this->dwDesc);
    checkCUDNN(cudnnSetFilter4dDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 1, 1));

    this->gradientOutputData = dnnLayer->allocateBlock(BERT::BATCH_SIZE * 1 * BERT::HIDDEN_SIZE * BERT::MAX_INPUT_LENGTH * sizeof(float));

    // Find out how much space we need for the workspace and allocate it
    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        *this->cudnnHandle,
        this->xDesc,
        this->outputTensorDescriptor,
        this->ffnConvolutionDescriptor,
        dwDesc,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        &this->backwardWorkspaceSize);
    this->backwardWorkspace = dnnLayer->allocateBlock(this->backwardWorkspaceSize);

    Profiler::pop();
}

void FeedForwardLayer::trainForward(cudnnTensorDescriptor_t inputTensorDescriptor, DNNLayer::Block* inputData) {

    dnnLayer->unspill();

    this->outputTensorDescriptor = inputTensorDescriptor;

    float alpha = 1;
    float beta = 1;

    Profiler::push("cudnnExecution");

    checkCUDNN(cudnnConvolutionForward(
        *this->cudnnHandle,
        &alpha,
        this->xDesc,
        this->convolutionInput.data.floats,
        this->wDesc,
        this->convolutionFilter.data.floats,
        this->ffnConvolutionDescriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        this->forwardWorkspace.data.floats,
        this->forwardWorkspaceSize,
        &beta,
        this->outputTensorDescriptor,
        this->outputTensorDeviceData.data.floats));
    
    spdlog::info("Feed forward output");
    this->outputTensorDeviceData.printData();

    Profiler::pop();
}

void FeedForwardLayer::trainBackward(cudnnTensorDescriptor_t inputGradientDescriptor, DNNLayer::Block* inputGradientData) {

    dnnLayer->unspill();

    float alpha = 1;
    float beta = 1;

    Profiler::push("cudnnExecution");

    checkCUDNN(cudnnConvolutionBackwardFilter(
        *this->cudnnHandle,
        &alpha,
        this->xDesc,
        this->convolutionInput.data.floats,
        inputGradientDescriptor,
        inputGradientData->data.floats,
        this->ffnConvolutionDescriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        this->backwardWorkspace.data.floats,
        this->backwardWorkspaceSize,
        &beta,
        dwDesc,
        this->gradientOutputData.data.floats));

    Profiler::pop();
}

cudnnTensorDescriptor_t FeedForwardLayer::getOutputTensorDescriptor() {
    return this->outputTensorDescriptor;
}

DNNLayer::Block* FeedForwardLayer::getOutputBlock() {
    return &this->outputTensorDeviceData;
}

cudnnTensorDescriptor_t FeedForwardLayer::getGradientDescriptor() {
    return this->outputTensorDescriptor;
}

DNNLayer::Block* FeedForwardLayer::getGradientBlock() {
    return &this->gradientOutputData;
}