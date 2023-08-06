#ifndef ADD_AND_NORMALIZE_LAYER_HPP
#define ADD_AND_NORMALIZE_LAYER_HPP

#include <cudnn.h>

#include "../include/memory_manager.hpp"

class NormalizationLayer {
public:
    NormalizationLayer(cudnnHandle_t* cudnnHandle, DNNLayer* dnnLayer);

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

    // Variables used in the forward and backwards training passes
    cudnnTensorDescriptor_t xDesc, normScaleBiasDesc, normMeanVarDesc, outputDescriptor;
    DNNLayer::Block *xData, normScaleData, normBiasData,
        dNormScaleData, dNormBiasData,
        runningMean, runningVariance, savedMean, savedInvVariance,
        dxData, outputData,
        forwardWorkspace, backwardWorkspace, reserveSpace;
    size_t forwardWorkspaceSizeBytes, backwardWorkspaceSizeBytes, reserveSpaceSizeBytes;
};

#endif