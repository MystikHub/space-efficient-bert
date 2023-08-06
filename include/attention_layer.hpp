#ifndef ATTENTION_LAYER_HPP
#define ATTENTION_LAYER_HPP

#include <string>

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas.h>

#include "../include/memory_manager.hpp"

using namespace std;

class AttentionLayer {
public:

    AttentionLayer(cudnnHandle_t* cudnnHandle, DNNLayer* dnnLayer);
    ~AttentionLayer();

    void trainForward(cudnnTensorDescriptor_t inputDescriptor, DNNLayer::Block* inputData);
    void trainBackward(cudnnTensorDescriptor_t inputGradientDescriptor, DNNLayer::Block* inputGradientData);
    void inferForward();

    cudnnTensorDescriptor_t getOutputTensorDescriptor();
    DNNLayer::Block* getOutputBlock();
    cudnnTensorDescriptor_t getGradientDescriptor();
    DNNLayer::Block* getGradientBlock();

private:
    cudnnHandle_t* cudnnHandle;
    DNNLayer* dnnLayer;

    // Data for the forward and backward passes
    cudnnAttnDescriptor_t attentionDescriptor;
    cudnnSeqDataDescriptor_t qDescriptor, kDescriptor, vDescriptor, oDescriptor;
    int *loWinIdx, *hiWinIdx;
    DNNLayer::Block attn_states, qData, kData, vData, weights,
        devSeqLengthsQO, devSeqLengthsKV,
        residualsDeviceData, workspaceDeviceData,
        reserveSpace, gradientDeviceData;
    size_t nFloats, reserveSpaceSizeInBytes, weightSizeInBytes, workSpaceSizeInBytes;

    cudnnTensorDescriptor_t outputTensorDescriptor;
    DNNLayer::Block outputTensorDeviceData;
};

#endif