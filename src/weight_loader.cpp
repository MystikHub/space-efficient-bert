#include "../include/weight_loader.hpp"

#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

#include "spdlog/spdlog.h"

#include "../include/cuda_utils.hpp"

using namespace std;

cudnnTensorDescriptor_t readWeights(string filepath) {
    
    ifstream dataFile(filepath, ios::binary);
    if(!dataFile.is_open()) {
        spdlog::error("Couldn't open file: {}", filepath);
        return NULL;
    }

    // Get the tensor dimensions
    unsigned int nchwInts[4] = {0, 0, 0, 0};
    std::filebuf* bufferReader = dataFile.rdbuf();

    char* result = new char[4 * sizeof(unsigned int)];
    bufferReader->sgetn(result, 4 * sizeof(unsigned int));
    memcpy(nchwInts, result, 4 * sizeof(unsigned int));

    // Read the data into a new CUDA tensor

    return NULL;
}

cudnnTensorDescriptor_t writeWeights(string filepath, unsigned int N, unsigned int C, unsigned int H, unsigned int W) {
    
    ifstream dataFile(filepath, ios::binary);

    unsigned int nchwInts[4] = {N, C, H, W};
    vector<unsigned char> writeData(4 * sizeof(unsigned int), ' ');
    memcpy(writeData.data(), nchwInts, 4 * sizeof(unsigned int));

    return NULL;
}

void fillRandom(float* startAddress, unsigned int count) {
    for(int i = 0; i < count; i++)
        startAddress[i] = rand() / (float) RAND_MAX;
}