#include "../include/attention_layer.hpp"

#include "../include/cuda_utils.hpp"
#include "../include/bert.hpp"
#include "../include/input_layer.hpp"
#include "../include/profiler.hpp"
#include "../include/weight_loader.hpp"

#include <string>

#define ENABLE_TENSOR_CORES

AttentionLayer::AttentionLayer(cudnnHandle_t* cudnnHandle, DNNLayer* dnnLayer)
    :cudnnHandle(cudnnHandle),
    dnnLayer(dnnLayer) {

    Profiler::push("cudaSetup");

    // CuDNN Attention descriptor
    this->attentionDescriptor = cudnnAttnDescriptor_t();
    checkCUDNN(cudnnCreateAttnDescriptor(&this->attentionDescriptor));
    size_t states_size;
    checkCUDNN(cudnnDropoutGetStatesSize(*this->cudnnHandle, &states_size));

    // Attn States
    this->attn_states = dnnLayer->allocateBlock(states_size);

    // Softmax dropout descriptor
    cudnnDropoutDescriptor_t softmaxDropoutDescriptor;
    checkCUDNN(cudnnCreateDropoutDescriptor(&softmaxDropoutDescriptor));
    checkCUDNN(cudnnSetDropoutDescriptor(
        softmaxDropoutDescriptor,
        *this->cudnnHandle,
        0.1,  // Dropout set according to the BERT paper
        attn_states.data.floats, // Not interested in dropout states
        states_size,    // No dropout state size
        0));   // Predictable dropout seed

    DNNLayer::Block post_states = dnnLayer->allocateBlock(states_size);

    cudnnDropoutDescriptor_t attentionDropoutDescriptor;
    checkCUDNN(cudnnCreateDropoutDescriptor(&attentionDropoutDescriptor));
    checkCUDNN(cudnnSetDropoutDescriptor(
        attentionDropoutDescriptor,
        *this->cudnnHandle,
        0.1,  // Dropout set according to the BERT paper
        post_states.data.floats, // Not interested in dropout states
        states_size,    // No dropout state size
        0));   // Predictable dropout seed

    checkCUDNN(cudnnSetAttnDescriptor(
        this->attentionDescriptor,
        0, // No extra attention parameters
        BERT::N_ATTENTION_HEADS,
        0, // No label smoothing; treat input data verbatim
        CUDNN_DATA_FLOAT,
        CUDNN_DATA_FLOAT,
#ifdef ENABLE_TENSOR_CORES
        CUDNN_TENSOR_OP_MATH,
#else
        CUDNN_DEFAULT_MATH,
#endif
        softmaxDropoutDescriptor, // Use the same dropout descriptor for the softmax input
        attentionDropoutDescriptor, //   and attention output
        BERT::HIDDEN_SIZE, // Q, K, and V embedding lengths
        BERT::HIDDEN_SIZE,
        BERT::HIDDEN_SIZE,
        BERT::HIDDEN_SIZE,                 // No projection for Q, K, V, and O
        BERT::HIDDEN_SIZE,
        BERT::HIDDEN_SIZE,
        BERT::HIDDEN_SIZE,
        BERT::MAX_INPUT_LENGTH, // Max sequence length for Q and K
        BERT::MAX_INPUT_LENGTH,
        BERT::BATCH_SIZE,
        1)); // Max beam size
    
    // Set up the low and high visibility indices. Bert is a _Bidirectional_
    // encoder, so it should see the words before and after the one being
    // processed
    this->loWinIdx = new int[BERT::MAX_INPUT_LENGTH];
    this->hiWinIdx = new int[BERT::MAX_INPUT_LENGTH];
    for(int i = 0; i < BERT::MAX_INPUT_LENGTH; i++) {
        loWinIdx[i] = 0;
        hiWinIdx[i] = BERT::MAX_INPUT_LENGTH;
    }
 
    // Sequence lengths of Q and O (for those annoying redundant parameters)
    this->devSeqLengthsQO = dnnLayer->allocateBlock(BERT::BATCH_SIZE * sizeof(int));
    this->devSeqLengthsQO.setupBuffer();
    for(int i = 0; i < BERT::BATCH_SIZE; i++)
        this->devSeqLengthsQO.hostBuffer.ints[i] = BERT::MAX_INPUT_LENGTH;
    this->devSeqLengthsQO.writeBuffer(true);

    // Same for K and V
    this->devSeqLengthsKV = dnnLayer->allocateBlock(BERT::BATCH_SIZE * sizeof(int));
    this->devSeqLengthsKV.setupBuffer();
    for(int i = 0; i < BERT::BATCH_SIZE; i++)
        this->devSeqLengthsKV.hostBuffer.ints[i] = BERT::MAX_INPUT_LENGTH;
    this->devSeqLengthsKV.writeBuffer(true);

    // Q, K, V, and O sequence descriptors
    this->qDescriptor = cudnnSeqDataDescriptor_t();
    this->kDescriptor = cudnnSeqDataDescriptor_t();
    this->vDescriptor = cudnnSeqDataDescriptor_t();
    this->oDescriptor = cudnnSeqDataDescriptor_t();
    checkCUDNN(cudnnCreateSeqDataDescriptor(&this->qDescriptor));
    checkCUDNN(cudnnCreateSeqDataDescriptor(&this->kDescriptor));
    checkCUDNN(cudnnCreateSeqDataDescriptor(&this->vDescriptor));
    checkCUDNN(cudnnCreateSeqDataDescriptor(&this->oDescriptor));

    // What is the shape of our sequence?
    int sequenceDataDescriptorDimA[CUDNN_SEQDATA_DIM_COUNT];
    sequenceDataDescriptorDimA[CUDNN_SEQDATA_TIME_DIM]  = BERT::MAX_INPUT_LENGTH;
    sequenceDataDescriptorDimA[CUDNN_SEQDATA_BATCH_DIM] = BERT::BATCH_SIZE;
    sequenceDataDescriptorDimA[CUDNN_SEQDATA_BEAM_DIM]  = 1; // Number of candidate results (i.e., pick the best of BEAM possible translations)
    sequenceDataDescriptorDimA[CUDNN_SEQDATA_VECT_DIM]  = BERT::HIDDEN_SIZE;

    // How many sequences per inference? BERT::BATCH_SIZE * BEAM
    // Create seqLengthArray (each element will be the full size)
    int seqLengthArray[BERT::BATCH_SIZE * 1];
    for(int i = 0; i < BERT::BATCH_SIZE * 1; i++)
        seqLengthArray[i] = BERT::MAX_INPUT_LENGTH;
    
    // What are the axes that will be traversed during attention?
    cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
    axes[3] = CUDNN_SEQDATA_VECT_DIM;
    axes[2] = CUDNN_SEQDATA_BEAM_DIM;
    axes[1] = CUDNN_SEQDATA_BATCH_DIM;
    axes[0] = CUDNN_SEQDATA_TIME_DIM;

    checkCUDNN(cudnnSetSeqDataDescriptor(
        this->qDescriptor,
        CUDNN_DATA_FLOAT,
        4, // Hard coded by nvidia (see https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetSeqDataDescriptor)
        sequenceDataDescriptorDimA,
        axes,
        BERT::BATCH_SIZE * 1,
        seqLengthArray,
        NULL)); // Only supported value for paddingFill

    checkCUDNN(cudnnSetSeqDataDescriptor(
        this->kDescriptor,
        CUDNN_DATA_FLOAT,
        4, // Hard coded by nvidia (see https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetSeqDataDescriptor)
        sequenceDataDescriptorDimA,
        axes,
        BERT::BATCH_SIZE * 1,
        seqLengthArray,
        NULL)); // Only supported value for paddingFill

    checkCUDNN(cudnnSetSeqDataDescriptor(
        this->vDescriptor,
        CUDNN_DATA_FLOAT,
        4, // Hard coded by nvidia (see https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetSeqDataDescriptor)
        sequenceDataDescriptorDimA,
        axes,
        BERT::BATCH_SIZE * 1,
        seqLengthArray,
        NULL)); // Only supported value for paddingFill

    checkCUDNN(cudnnSetSeqDataDescriptor(
        this->oDescriptor,
        CUDNN_DATA_FLOAT,
        4, // Hard coded by nvidia (see https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetSeqDataDescriptor)
        sequenceDataDescriptorDimA,
        axes,
        BERT::BATCH_SIZE * 1,
        seqLengthArray,
        NULL)); // Only supported value for paddingFill

    // Allocate Q, K, V, and O in device memory (NCHW)
    // Use random queries, keys, and values for proof-of-concept
    this->nFloats = BERT::BATCH_SIZE * 1 * BERT::HIDDEN_SIZE * BERT::MAX_INPUT_LENGTH * BERT::N_ATTENTION_HEADS;
    size_t nBytes = this->nFloats * sizeof(float);

    this->qData = dnnLayer->allocateBlock(nBytes);
    this->kData = dnnLayer->allocateBlock(nBytes);
    this->vData = dnnLayer->allocateBlock(nBytes);
    this->outputTensorDeviceData = dnnLayer->allocateBlock(nBytes);

    this->qData.loadRandom();
    this->kData.loadRandom();
    this->vData.loadRandom();
    this->outputTensorDeviceData.loadRandom();

    this->residualsDeviceData = dnnLayer->allocateBlock(nBytes);

    // Set up and pass in the weights
    checkCUDNN(cudnnGetMultiHeadAttnBuffers(
        *this->cudnnHandle,
        this->attentionDescriptor,
        &this->weightSizeInBytes,
        &this->workSpaceSizeInBytes,
        &this->reserveSpaceSizeInBytes
    ));

    // We need weights for Q, K, V, for each head
    this->weights = dnnLayer->allocateBlock(this->weightSizeInBytes);
    this->weights.loadRandom();

    // Temporary workspace allocation
    this->workspaceDeviceData = dnnLayer->allocateBlock(this->workSpaceSizeInBytes);
    // Reserve space for storing gradients in backpropagation
    this->reserveSpace = dnnLayer->allocateBlock(this->reserveSpaceSizeInBytes);
    this->gradientDeviceData = dnnLayer->allocateBlock(this->nFloats);

    this->outputTensorDescriptor = cudnnTensorDescriptor_t();
    checkCUDNN(cudnnCreateTensorDescriptor(&this->outputTensorDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->outputTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BERT::BATCH_SIZE, 1, BERT::HIDDEN_SIZE, BERT::MAX_INPUT_LENGTH));

    Profiler::pop();
}

AttentionLayer::~AttentionLayer() {

    Profiler::push("memoryManagement");

    checkCudaErrors(cudaFree(this->outputTensorDeviceData.data.floats));
    checkCUDNN(cudnnDestroyTensorDescriptor(this->outputTensorDescriptor));

    Profiler::pop();
}

void AttentionLayer::trainForward(cudnnTensorDescriptor_t inputDescriptor, DNNLayer::Block* inputData) {

    dnnLayer->unspill();

    Profiler::push("cudnnExecution");

    // Execute the multi-head attention layer
    // Weight gradients will be calculated during backpropagation
    checkCUDNN(cudnnMultiHeadAttnForward(
        *this->cudnnHandle,
        this->attentionDescriptor,
        -1, // Current index = -1 (go through all input indices)
        loWinIdx,
        hiWinIdx,
        this->devSeqLengthsQO.data.ints,
        this->devSeqLengthsKV.data.ints,
        this->qDescriptor,
        this->qData.data.floats,
        this->residualsDeviceData.data.floats,
        this->kDescriptor,
        this->kData.data.floats,
        this->vDescriptor,
        this->vData.data.floats,
        this->oDescriptor,
        this->outputTensorDeviceData.data.floats,
        this->weightSizeInBytes, // Weight size
        this->weights.data.floats,
        this->workSpaceSizeInBytes, // Workspace size in bytes
        this->workspaceDeviceData.data.floats,
        this->reserveSpaceSizeInBytes,
        this->reserveSpace.data.floats));       // Probably need to preserve the data in here for the backward pass
                                    // The documentation doesn't mention this, but other functions need it,
                                    // so keep it to be safe
    
    Profiler::pop();
    
    spdlog::info("Attention output:");
    this->outputTensorDeviceData.printData();
}

void AttentionLayer::trainBackward(cudnnTensorDescriptor_t dyDesc, DNNLayer::Block* dyDeviceData) {

    dnnLayer->unspill();

    Profiler::push("cudnnExecution");

    checkCUDNN(cudnnMultiHeadAttnBackwardWeights(
        *this->cudnnHandle,
        this->attentionDescriptor,
        CUDNN_WGRAD_MODE_SET,          // How to update the weights in different batches; by updating them directly
        this->qDescriptor,
        this->qData.data.floats,
        this->kDescriptor,
        this->kData.data.floats,
        this->vDescriptor,
        this->vData.data.floats,
        this->oDescriptor,
        dyDeviceData->data.floats,
        this->weightSizeInBytes,
        this->weights.data.floats,
        this->gradientDeviceData.data.floats,
        this->workSpaceSizeInBytes,
        this->workspaceDeviceData.data.floats,
        this->reserveSpaceSizeInBytes,
        this->reserveSpace.data.floats));
    
    Profiler::pop();
}

cudnnTensorDescriptor_t AttentionLayer::getOutputTensorDescriptor() {
    return this->outputTensorDescriptor;
}

DNNLayer::Block* AttentionLayer::getOutputBlock() {
    return &this->outputTensorDeviceData;
}

cudnnTensorDescriptor_t AttentionLayer::getGradientDescriptor() {
    return this->outputTensorDescriptor;
}

DNNLayer::Block* AttentionLayer::getGradientBlock() {
    return &this->gradientDeviceData;
}