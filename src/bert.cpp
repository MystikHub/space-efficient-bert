#include "spdlog/spdlog.h"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_ops_infer.h>

#include "../include/bert.hpp"
#include "../include/cuda_utils.hpp"
#include "../include/memory_manager.hpp"
#include "../include/profiler.hpp"
#include "../include/token_reader.hpp"
#include "../include/weight_loader.hpp"

BERT::BERT(size_t maxMemoryInBytes, bool enableMemoryOptimizations)
    : manager(MemoryManager(maxMemoryInBytes, enableMemoryOptimizations)) {

    Profiler::push("cudaSetup");
    checkCUDNN(cudnnCreate(&this->cudnnHandle));
    Profiler::pop();
}

void BERT::train(TokenReader trainingTokens, unsigned int nEpochs) {
    spdlog::info("Starting BERT training");
    spdlog::info("Creating a TokenReader using file: {}", trainingTokens.getFilePath());

    DNNLayer* inputLayerData = manager.getNextActiveLayer();
    InputLayer inputLayer(&this->cudnnHandle, trainingTokens.getFilePath(), inputLayerData);
    DNNLayer* outputLayer = NULL;
    
    // Initialize encoder layers
    for(int i = 0; i < BERT::N_LAYERS; i++) {
        this->encoderLayers.push_back(new EncoderLayer(&this->cudnnHandle, i, &this->manager));
        this->manager.reportUsage();
    }

    for(int epochIteration = 0; epochIteration < nEpochs; epochIteration++) {

        // Only use one input per epoch for now
        for(; !trainingTokens.reachedEnd(); trainingTokens++) {

            // Forward the data through each encoder layer
            for(int i = 0; i < this->encoderLayers.size(); i++) {
                EncoderLayer* encoderLayer = this->encoderLayers[i];

                if(i == 0) {
                    // Create random input embeddings
                    inputLayer.trainForward(*trainingTokens);
                    
                    cudnnTensorDescriptor_t initialTensorDescriptor = inputLayer.getOutputTensorDescriptor();
                    DNNLayer::Block* initialTensorBlock = inputLayer.getOutputBlock();

                    encoderLayer->trainForward(initialTensorDescriptor, initialTensorBlock);
                } else {
                    encoderLayer->trainForward(this->encoderLayers[i - 1]->getOutputTensorDescriptor(), this->encoderLayers[i - 1]->getOutputBlock());
                }
            }

            // Set up data management for the output layer ONLY AFTER all the
            // transformer layers have been set up
            if(outputLayer == NULL) {
                outputLayer = manager.getNextActiveLayer();
            }

            // Instruct the memory manager that we need memory in the other
            // direction
            manager.changeDirections();

            // Update the weights on a backwards pass
            // Use a random gradient tensor to take the place of the result of the cost function
            for(int i = this->encoderLayers.size() - 1; i >= 0; i--) {
                EncoderLayer* encoderLayer = this->encoderLayers[i];

                if(i == this->encoderLayers.size() - 1) {
                    cudnnTensorDescriptor_t initialGradientsDescriptor = this->encoderLayers[i]->getOutputTensorDescriptor();
                    size_t nGradientElements = BERT::BATCH_SIZE * BERT::HIDDEN_SIZE * BERT::BATCH_SIZE;

                    DNNLayer::Block initialGradientsBlock = outputLayer->allocateBlock(nGradientElements * sizeof(float));
                    initialGradientsBlock.loadRandom();

                    encoderLayer->trainBackward(initialGradientsDescriptor, &initialGradientsBlock);
                } else {
                    encoderLayer->trainBackward(this->encoderLayers[i + 1]->getGradientDescriptor(), this->encoderLayers[i + 1]->getGradientBlock());
                }
            }

            manager.changeDirections();
        }

        trainingTokens.reset();
        spdlog::info("Trained {} epochs", epochIteration + 1);
    }
}

void BERT::infer(TokenReader trainingTokens) {
    spdlog::info("Starting BERT inference");
    spdlog::info("Creating a TokenReader using file: {}", trainingTokens.getFilePath());
}