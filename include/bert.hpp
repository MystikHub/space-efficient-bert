#ifndef BERT_HPP
#define BERT_HPP

#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "encoder_layer.hpp"
#include "input_layer.hpp"
#include "memory_manager.hpp"
#include "token_reader.hpp"

class BERT {
public:
    static const unsigned int VOCAB_SIZE = 30522;

#define BERT_BASE
#ifdef BERT_LARGE
    static const unsigned int BATCH_SIZE = 4;
    static const unsigned int HIDDEN_SIZE = 1024;
    static const unsigned int MAX_INPUT_LENGTH = 128;
    static const unsigned int N_LAYERS = 24;
    static const unsigned int N_ATTENTION_HEADS = 16;
#endif
#ifdef BERT_BASE
    static const unsigned int BATCH_SIZE = 4;
    static const unsigned int HIDDEN_SIZE = 768;
    static const unsigned int MAX_INPUT_LENGTH = 128;
    static const unsigned int N_LAYERS = 12;
    static const unsigned int N_ATTENTION_HEADS = 12;
#endif
#ifdef BERT_SMALL
    static const unsigned int BATCH_SIZE = 2;
    static const unsigned int HIDDEN_SIZE = 512;
    static const unsigned int MAX_INPUT_LENGTH = 128;
    static const unsigned int N_LAYERS = 4;
    static const unsigned int N_ATTENTION_HEADS = 4;
#endif

    BERT(size_t maxMemoryInBytes, bool enableMemoryOptimizations);

    void train(TokenReader trainingTokens, unsigned int nEpochs);
    void infer(TokenReader inferenceTokens);

private:
    MemoryManager manager;

    InputLayer inputLayer;
    std::vector<EncoderLayer*> encoderLayers;

    cudnnHandle_t cudnnHandle;
};

#endif