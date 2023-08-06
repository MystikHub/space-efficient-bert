#include "../include/encoder_layer.hpp"
#include "../include/memory_manager.hpp"

#include <cudnn.h>
#include <spdlog/spdlog.h>

#include "../include/normalization_layer.hpp"
#include "cuda_utils.hpp"

EncoderLayer::EncoderLayer(cudnnHandle_t* cudnnHandle, int transformerIndex, MemoryManager* bertManager)
    : cudnnHandle(cudnnHandle),
    manager(bertManager),
    multiHeadAttention(cudnnHandle, bertManager->getNextActiveLayer()),
    addNormalize1(cudnnHandle, bertManager->getNextActiveLayer()),
    feedForward(cudnnHandle, bertManager->getNextActiveLayer()),
    addNormalize2(cudnnHandle, bertManager->getNextActiveLayer()),
    transformerIndex(transformerIndex) {

}

void EncoderLayer::trainForward(cudnnTensorDescriptor_t inputTensorDescriptor, DNNLayer::Block* inputData) {
    spdlog::info("Encoder {} input tensor:", transformerIndex);
    inputData->printData();

    this->multiHeadAttention.trainForward(inputTensorDescriptor, inputData);
    manager->reportUsage();
    this->addNormalize1.trainForward(this->multiHeadAttention.getOutputTensorDescriptor(), this->multiHeadAttention.getOutputBlock());
    manager->reportUsage();

    this->feedForward.trainForward(this->addNormalize1.getOutputTensorDescriptor(), this->addNormalize1.getOutputBlock());
    manager->reportUsage();
    this->addNormalize2.trainForward(this->feedForward.getOutputTensorDescriptor(), this->feedForward.getOutputBlock());
    manager->reportUsage();
}

void EncoderLayer::trainBackward(cudnnTensorDescriptor_t outputGradientDesc, DNNLayer::Block* outputGradientData) {
    spdlog::info("Updating weights on encoder {}", transformerIndex);

    this->addNormalize2.trainBackward(outputGradientDesc, outputGradientData);
    manager->reportUsage();
    this->feedForward.trainBackward(this->addNormalize2.getGradientDescriptor(), this->addNormalize2.getGradientBlock());
    manager->reportUsage();

    this->addNormalize1.trainBackward(this->feedForward.getGradientDescriptor(), this->feedForward.getGradientBlock());
    manager->reportUsage();
    this->multiHeadAttention.trainBackward(this->addNormalize1.getGradientDescriptor(), this->addNormalize1.getGradientBlock());
    manager->reportUsage();
}

cudnnTensorDescriptor_t EncoderLayer::getOutputTensorDescriptor() {
    return this->addNormalize2.getOutputTensorDescriptor();
}

DNNLayer::Block* EncoderLayer::getOutputBlock() {
    return this->addNormalize2.getOutputBlock();
}

cudnnTensorDescriptor_t EncoderLayer::getGradientDescriptor() {
    return this->multiHeadAttention.getGradientDescriptor();
}

DNNLayer::Block* EncoderLayer::getGradientBlock() {
    return this->multiHeadAttention.getGradientBlock();
}