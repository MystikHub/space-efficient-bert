#include "../include/bert.hpp"
#include "../include/cuda_utils.hpp"
#include "../include/memory_manager.hpp"
#include "../include/normalization_layer.hpp"
#include "../include/profiler.hpp"
#include "../include/weight_loader.hpp"

#include <cudnn_ops_infer.h>

NormalizationLayer::NormalizationLayer(cudnnHandle_t* cudnnHandle, DNNLayer* dnnLayer)
    : cudnnHandle(cudnnHandle),
    dnnLayer(dnnLayer) {

    Profiler::push("cudaSetup");
    
    size_t nLayerElements = BERT::BATCH_SIZE * 1 * BERT::HIDDEN_SIZE * BERT::MAX_INPUT_LENGTH;
    size_t nLayerElementsSize = nLayerElements * sizeof(float);

    // For per-activation batch normalization, the scale, biases, mean, and variances should be the same size as the input
    this->xDesc                      = cudnnTensorDescriptor_t();
    this->outputDescriptor           = cudnnTensorDescriptor_t();
    this->normScaleBiasDesc          = cudnnTensorDescriptor_t();
    this->normMeanVarDesc            = cudnnTensorDescriptor_t();

    checkCUDNN(cudnnCreateTensorDescriptor(&this->xDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&this->outputDescriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&this->normScaleBiasDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&this->normMeanVarDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->xDesc,             CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BERT::BATCH_SIZE, 1, BERT::HIDDEN_SIZE, BERT::MAX_INPUT_LENGTH));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->outputDescriptor,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BERT::BATCH_SIZE, 1, BERT::HIDDEN_SIZE, BERT::MAX_INPUT_LENGTH));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->normScaleBiasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BERT::BATCH_SIZE, 1, BERT::HIDDEN_SIZE, BERT::MAX_INPUT_LENGTH));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->normMeanVarDesc,   CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BERT::BATCH_SIZE, 1, BERT::HIDDEN_SIZE, BERT::MAX_INPUT_LENGTH));

    this->normScaleData    = dnnLayer->allocateBlock(nLayerElementsSize);
    this->normBiasData     = dnnLayer->allocateBlock(nLayerElementsSize);
    this->runningMean      = dnnLayer->allocateBlock(nLayerElementsSize);
    this->runningVariance  = dnnLayer->allocateBlock(nLayerElementsSize);
    this->savedMean        = dnnLayer->allocateBlock(nLayerElementsSize);
    this->savedInvVariance = dnnLayer->allocateBlock(nLayerElementsSize);
    this->outputData       = dnnLayer->allocateBlock(nLayerElementsSize);

    normScaleData.loadRandom();
    normBiasData.loadRandom();
    runningMean.loadRandom();
    runningVariance.loadRandom();
    outputData.loadRandom();

    checkCUDNN(cudnnGetNormalizationForwardTrainingWorkspaceSize(
        *this->cudnnHandle,
            CUDNN_NORM_PER_ACTIVATION,
            CUDNN_NORM_OPS_NORM,
            CUDNN_NORM_ALGO_STANDARD,
        this->xDesc,
            NULL, // zDesc,
            this->outputDescriptor,
            this->normScaleBiasDesc,
            NULL, // activationDesc
            this->normMeanVarDesc,
            &this->forwardWorkspaceSizeBytes,
            1));
    this->forwardWorkspace = dnnLayer->allocateBlock(forwardWorkspaceSizeBytes);

    checkCUDNN(cudnnGetNormalizationBackwardWorkspaceSize(
        *this->cudnnHandle,
            CUDNN_NORM_PER_ACTIVATION,
            CUDNN_NORM_OPS_NORM,
            CUDNN_NORM_ALGO_STANDARD,
        this->xDesc,
        this->outputDescriptor, // yDesc
        this->outputDescriptor, // dyDesc
        NULL, // dzDesc
        this->outputDescriptor, // dxDesc
        this->normScaleBiasDesc, // dNormScaleBiasDesc
        NULL, // activationDesc,
        this->normMeanVarDesc,
        &this->backwardWorkspaceSizeBytes,
        1));
    this->backwardWorkspace = dnnLayer->allocateBlock(this->backwardWorkspaceSizeBytes);

    checkCUDNN(cudnnGetNormalizationTrainingReserveSpaceSize(
        *this->cudnnHandle,
            CUDNN_NORM_PER_ACTIVATION,
            CUDNN_NORM_OPS_NORM,
            CUDNN_NORM_ALGO_STANDARD,
            NULL,
        this->xDesc,
            &this->reserveSpaceSizeBytes,
            1));
    this->reserveSpace = dnnLayer->allocateBlock(this->reserveSpaceSizeBytes);

    this->dxData      = dnnLayer->allocateBlock(nLayerElementsSize);
    this->dNormScaleData    = dnnLayer->allocateBlock(nLayerElementsSize);
    this->dNormBiasData     = dnnLayer->allocateBlock(nLayerElementsSize);

    Profiler::pop();
}

void NormalizationLayer::trainForward(
    cudnnTensorDescriptor_t attentionOutputDescriptor,
    DNNLayer::Block* attentionOutputData) {

    dnnLayer->unspill();

    float alpha = 1;
    float beta  = 1;

    this->xData = attentionOutputData;

    Profiler::push("cudnnExecution");

    checkCUDNN(cudnnNormalizationForwardTraining(
        *this->cudnnHandle,
            CUDNN_NORM_PER_ACTIVATION,
            CUDNN_NORM_OPS_NORM,
            CUDNN_NORM_ALGO_STANDARD,
        &alpha,
        &beta,
        this->xDesc,
            this->xData->data.floats,
        this->normScaleBiasDesc,
        this->normScaleData.data.floats,      // Trainable parameters, need to keep these in memory
        this->normBiasData.data.floats,
            1.0f,
        this->normMeanVarDesc,
        this->runningMean.data.floats,
        this->runningVariance.data.floats,
            1e-6,
        this->savedMean.data.floats,
        this->savedInvVariance.data.floats,
            NULL,                       // Activation descriptor
        NULL,                           // Parameters for Z sum
        NULL, 
        this->outputDescriptor,
        this->outputData.data.floats,
        this->forwardWorkspace.data.floats,
        this->forwardWorkspaceSizeBytes,
        this->reserveSpace.data.floats,       // This data needs to be preserved for the backwards pass
        this->reserveSpaceSizeBytes,
        1));

    spdlog::info("Normalization output:");
    this->outputData.printData();

    Profiler::pop();
}

void NormalizationLayer::trainBackward(cudnnTensorDescriptor_t dyDesc, DNNLayer::Block* dyDeviceData) {

    dnnLayer->unspill();

    float alphaDataDiff  = 1;
    float betaDataDiff   = 1;
    float alphaParamDiff = 1;
    float betaParamDiff  = 1;

    Profiler::push("cudnnExecution");

    checkCUDNN(cudnnNormalizationBackward(
        *this->cudnnHandle,
            CUDNN_NORM_PER_ACTIVATION,  // Same as during training
            CUDNN_NORM_OPS_NORM,
            CUDNN_NORM_ALGO_STANDARD,
            &alphaDataDiff,              // Scaling factors to "blend" dx (sounds like the
            &betaDataDiff,               // learning rate in SGD)
            &alphaParamDiff,             // Same but for NormScale...
            &betaParamDiff,              // and normBias
            this->xDesc,
            this->xData->data.floats,
            NULL,                       // yDesc and yData can be set to null       
            NULL,                       //    if the CUDNN_NORM_OPS_NORM is set above
            dyDesc,
            dyDeviceData->data.floats,
        NULL,                           // dz desc and data, again not needed
            NULL,                       //     if CUDNN_NORM_OPS_NORM is set
            this->outputDescriptor,// dx descriptor
            this->dxData.data.floats,
            this->normScaleBiasDesc,   // dNormScaleBiasDesc
            this->normScaleData.data.floats,
            this->normBiasData.data.floats,
            this->dNormScaleData.data.floats,
            this->dNormBiasData.data.floats,
            1e-6,
            this->normMeanVarDesc,
            this->savedMean.data.floats,
            this->savedInvVariance.data.floats,
            NULL,                       // Not needed for CUDNN_NORM_OPS_NORM
            this->backwardWorkspace.data.floats,
            this->backwardWorkspaceSizeBytes,
            this->reserveSpace.data.floats,
            this->reserveSpaceSizeBytes,
        1));

    Profiler::pop();
}

cudnnTensorDescriptor_t NormalizationLayer::getOutputTensorDescriptor() {
    return this->outputDescriptor;
}

DNNLayer::Block* NormalizationLayer::getOutputBlock() {
    return &this->outputData;
}

cudnnTensorDescriptor_t NormalizationLayer::getGradientDescriptor() {
    return this->outputDescriptor;
}

DNNLayer::Block* NormalizationLayer::getGradientBlock() {
    return &this->dxData;
}