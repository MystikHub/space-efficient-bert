#ifndef ENCODER_LAYER_HPP
#define ENCODER_LAYER_HPP

#include <cudnn.h>
#include <vector>

#include "attention_layer.hpp"
#include "feed_forward_layer.hpp"
#include "memory_manager.hpp"
#include "normalization_layer.hpp"

class EncoderLayer {
public:
    EncoderLayer(cudnnHandle_t* cudnnHandle, int transformerIndex, MemoryManager* bertManager);

    void trainForward(cudnnTensorDescriptor_t inputTensorDescriptor, DNNLayer::Block* inputData);
    void trainBackward(cudnnTensorDescriptor_t outputGradientDesc, DNNLayer::Block* outputGradientData);
    void inferForward();

    int transformerIndex;

    AttentionLayer multiHeadAttention;
    NormalizationLayer addNormalize1;
    FeedForwardLayer feedForward;
    NormalizationLayer addNormalize2;

    cudnnTensorDescriptor_t getOutputTensorDescriptor();
    DNNLayer::Block* getOutputBlock();
    cudnnTensorDescriptor_t getGradientDescriptor();
    DNNLayer::Block* getGradientBlock();

private:
    cudnnHandle_t* cudnnHandle;
    MemoryManager* manager;

    cudnnTensorDescriptor_t outputTensorDescriptor;
};

#endif