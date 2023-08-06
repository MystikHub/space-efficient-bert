#ifndef FEED_FORWARD_LAYER_HPP
#define FEED_FORWARD_LAYER_HPP

#include <cudnn.h>

#include "../include/memory_manager.hpp"

class FeedForwardLayer {
public:
    FeedForwardLayer(cudnnHandle_t* cudnnHandle, DNNLayer* dnnLayer);

    void trainForward(cudnnTensorDescriptor_t inputTensorDescriptor, DNNLayer::Block* inputData);
    void trainBackward(cudnnTensorDescriptor_t dyDesc, DNNLayer::Block* dyDeviceData);

    void inferForward();

    cudnnTensorDescriptor_t getOutputTensorDescriptor();
    DNNLayer::Block* getOutputBlock();
    cudnnTensorDescriptor_t getGradientDescriptor();
    DNNLayer::Block* getGradientBlock();

private:
    cudnnHandle_t* cudnnHandle;
    DNNLayer* dnnLayer;

    cudnnConvolutionDescriptor_t ffnConvolutionDescriptor;
    cudnnTensorDescriptor_t xDesc, outputTensorDescriptor;
    cudnnFilterDescriptor_t wDesc, dwDesc;
    DNNLayer::Block convolutionInput, convolutionFilter, gradientOutputData, outputTensorDeviceData, forwardWorkspace, backwardWorkspace;
    size_t forwardWorkspaceSize, backwardWorkspaceSize;
};

#endif