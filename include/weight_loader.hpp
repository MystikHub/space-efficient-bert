#ifndef WEIGHT_LOADER_HPP
#define WEIGHT_LOADER_HPP

#include <string>

#include <cuda_runtime.h>
#include <cudnn.h>

using namespace std;

cudnnTensorDescriptor_t readWeights(string filepath);
cudnnTensorDescriptor_t writeWeights(string filepath, unsigned int N, unsigned int C, unsigned int H, unsigned int W);

void fillRandom(float* startAddress, unsigned int count);

void store_weights(string filepath, cudnnTensorDescriptor_t* tensor);


#endif