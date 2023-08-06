#ifndef MEMORY_MANAGER_HPP
#define MEMORY_MANAGER_HPP

#include <vector>

#define PRINT_INFO

using namespace std;

class MemoryManager;

class DNNLayer {
public:
    struct Block {
        union storageUnion {
            float* floats;
            int* ints;
        } data, hostBuffer;

        size_t size;

        void loadRandom();

        void setupBuffer();
        void readBuffer();
        void writeBuffer(bool deleteOnWrite);
        void deleteBuffer();

        void printData();
    };

    DNNLayer(unsigned int layerIndex, float* startAddress, float* memoryPoolUpperLimit, MemoryManager* manager);
    // ~DNNLayer();

    size_t getLayerSizeInWords();

    Block allocateBlock(size_t nBytes);
    void spill();
    void unspill();
    void moveDeviceData(float* newDeviceStartAddress);

    float* startAddress;
    vector<Block> blocks;
    MemoryManager* manager;
    
    int age;
    bool spilled;
    unsigned int index;

private:
    float* memoryPoolUpperLimit;
    float* spilledDataInSystemMemory;
};

class MemoryManager {
public:
    MemoryManager(size_t poolSize, bool enableOptimizations);

    bool enableOptimizations;

    float* pool;
    size_t poolSizeInWords;

    vector<DNNLayer*> layers;

    enum trainingDirection {
        forwards,
        backwards
    } direction = forwards;

    void reallocateLayer(unsigned int layerIndex, size_t newSizeInWords);
    float* makeRoom(size_t requiredSpaceInWords);

    // Alternates the memory location to write to i.e. if previously writing to
    // the bottom, spill the top and begin writing there
    void changeDirections();
    void reportUsage();

    DNNLayer* getNextActiveLayer();
    void ageLayers();

private:
    DNNLayer* getNewestAllocatedLayer();
    DNNLayer* getOldestAllocatedLayer();
};

#endif