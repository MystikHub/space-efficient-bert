#include "../include/cuda_utils.hpp"
#include "../include/memory_manager.hpp"
#include "../include/profiler.hpp"
#include "../include/weight_loader.hpp"

MemoryManager::MemoryManager(size_t poolSizeInBytes, bool enableOptimizations) {
    Profiler::push("memoryManagement");

    this->poolSizeInWords = poolSizeInBytes / 4;
    this->enableOptimizations = enableOptimizations;

    if(enableOptimizations)
        checkCudaErrors(cudaMalloc(&this->pool, poolSizeInBytes));

    Profiler::pop();
}

DNNLayer::DNNLayer(unsigned int layerIndex, float* startAddress, float* memoryPoolUpperLimit, MemoryManager* manager)
    :index(layerIndex),
    startAddress(startAddress),
    memoryPoolUpperLimit(memoryPoolUpperLimit),
    manager(manager) {

    Profiler::push("memoryManagement");

    this->age = 0;
    this->spilled = false;

    Profiler::pop();
}

size_t DNNLayer::getLayerSizeInWords() {

    Profiler::push("memoryManagement");

    size_t layerSize = 0;
    for(int i = 0; i < this->blocks.size(); i++) {
        layerSize += this->blocks[i].size;
    }

    Profiler::pop();
    return layerSize;
}

DNNLayer::Block DNNLayer::allocateBlock(size_t nBytes) {

    Profiler::push("memoryManagement");

    size_t wordsUsedByLayerBlocks = getLayerSizeInWords();
    float* newBlockStartAddress = this->startAddress + wordsUsedByLayerBlocks;

    size_t nWords = nBytes / 4;
    float* newBlockEndAddress = newBlockStartAddress + nWords;

    // In unoptimized mode, make a regular call to cudaMalloc
    if(!manager->enableOptimizations) {
        checkCudaErrors(cudaMalloc(&newBlockStartAddress, nBytes));

    // Otherwise, find a suitable place within the memory pool
    } else {

        // Check if the end of this block crosses the memory pool limit...
        bool invalidStartAddress = false;
        if(newBlockEndAddress > memoryPoolUpperLimit) {
            invalidStartAddress = true;
        }

        // or enters another DNN layer
        if(manager->enableOptimizations) {
            for(int i = 0; i < manager->layers.size(); i++) {
                float* layerStart = manager->layers[i]->startAddress;
                float* layerEnd = layerStart + manager->layers[i]->getLayerSizeInWords();

                if(newBlockEndAddress > layerStart
                    && newBlockEndAddress < layerEnd
                    && !manager->layers[i]->spilled) {
                    invalidStartAddress = true;
                    break;
                }
            }
        }

        if(invalidStartAddress) {
            spdlog::warn("Tried allocating past the end of the manager's pool");

            if(manager->enableOptimizations) {
                size_t freeSpace = (wordsUsedByLayerBlocks + nWords) * 2;
                spdlog::info("Asking the memory manager to make room for {} bytes", freeSpace * 4);

                manager->reallocateLayer(this->index, freeSpace);
                // This layer's start address likely changed during the previous
                // call, so recalculate it
                newBlockStartAddress = this->startAddress + wordsUsedByLayerBlocks;
            } else {
                spdlog::error("Please allocate more memory for the memory manager");
                exit(1);
            }
        }
    }

    Block newBlock;
    newBlock.data.floats = newBlockStartAddress;
    newBlock.size = nWords;

    this->blocks.push_back(newBlock);

    Profiler::pop();
    return newBlock;
}

void DNNLayer::spill() {

    if(!manager->enableOptimizations)
        return;
    Profiler::push("memorySpill");

    this->spilled = true;
    this->age = -1;

    // All blocks in each layer are allocated contiguously, so we can copy it
    // all in one operation
    size_t layerSizeInWords = getLayerSizeInWords();
    size_t layerSizeInBytes = layerSizeInWords * 4;
    spilledDataInSystemMemory = new float[layerSizeInWords];
    checkCudaErrors(cudaMemcpy(spilledDataInSystemMemory, startAddress, layerSizeInWords, cudaMemcpyDeviceToHost));

    // Be safe and set each block's pointers to NULL
    for(int i = 0; i < blocks.size(); i++)
        blocks[i].data.floats = NULL;
    
    spdlog::info("Spilled layer {}", index);
    manager->reportUsage();

    Profiler::pop();
}

// Ensure that the data for this layer is loaded onto the device
void DNNLayer::unspill() {

    if(!manager->enableOptimizations)
        return;
    Profiler::push("memoryUnspill");

    if(manager->enableOptimizations) {

        // This function is called at the beginning of each layer's forwards or
        // backwards pass, regardless of whether it's spilled or not
        if(spilled) {
            size_t layerSizeInWords = getLayerSizeInWords();

            spdlog::info("Unspilling layer {}", index);
            spdlog::info("Asking the memory manager to make space for {} MB", layerSizeInWords * 4e-6);
            spdlog::info("Current memory manager state:");
            manager->reportUsage();

            float* newLayerLocation = manager->makeRoom(layerSizeInWords);

            // Unspill the data into the new location
            checkCudaErrors(cudaMemcpy(newLayerLocation, spilledDataInSystemMemory, layerSizeInWords, cudaMemcpyHostToDevice));
            delete spilledDataInSystemMemory;

            this->startAddress = newLayerLocation;
            this->spilled = false;

            spdlog::info("Unspilled layer {}", index);
        }

        // This has become the newest allocated layer
        manager->ageLayers();
        this->age = 0;
    }

    Profiler::pop();
}

// Called during reallocation
void DNNLayer::moveDeviceData(float* newDeviceStartAddress) {

    Profiler::push("memoryMoveDeviceData");

    size_t layerSize = getLayerSizeInWords();
    checkCudaErrors(cudaMemcpy(newDeviceStartAddress, this->startAddress, layerSize, cudaMemcpyDeviceToDevice));

    this->startAddress = newDeviceStartAddress;

    // Update all the block pointers
    float* newBlockStartAddress = this->startAddress;
    for(int i = 0; i < blocks.size(); i++) {
        blocks[i].data.floats = newBlockStartAddress;
        newBlockStartAddress += blocks[i].size;
    }

    Profiler::pop();
}

void DNNLayer::Block::loadRandom() {

    Profiler::push("randomNumberGeneration");

    // Create a random array in system memory first
    float* hostData = new float[size];
    fillRandom(hostData, size);

    Profiler::pop();
    Profiler::push("memoryLoadRandom");

    // No allocation needed, just copy it and update our size
    checkCudaErrors(cudaMemcpy(this->data.floats, hostData, size, cudaMemcpyHostToDevice));

    delete hostData;

    Profiler::pop();
}

void DNNLayer::Block::setupBuffer() {

    Profiler::push("memoryManagement");

    this->hostBuffer.floats = new float[size / sizeof(float)];

    Profiler::pop();
}

void DNNLayer::Block::readBuffer() {

    Profiler::push("memoryReadFromDevice");

    checkCudaErrors(cudaMemcpy(hostBuffer.floats, data.floats, size, cudaMemcpyDeviceToHost));

    Profiler::pop();
}

void DNNLayer::Block::writeBuffer(bool deleteOnWrite) {

    Profiler::push("memoryManagement");

    checkCudaErrors(cudaMemcpy(data.floats, hostBuffer.floats, size, cudaMemcpyHostToDevice));

    if(deleteOnWrite) {
        this->deleteBuffer();
    }

    Profiler::pop();
}

void DNNLayer::Block::deleteBuffer() {
    delete this->hostBuffer.floats;
}

void DNNLayer::Block::printData() {

    Profiler::push("messages");

#ifdef PRINT_INFO
    // Global tensor data visibility toggle
    // Comment or uncomment the return statement below as needed
    // return;

    this->setupBuffer();
    this->readBuffer();

    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%.3f ", hostBuffer.floats[i * 4 + j]);
        }
        printf("\n");
    }
#endif

    Profiler::pop();
}

void MemoryManager::reallocateLayer(unsigned int layerIndex, size_t newSizeInWords) {

    Profiler::push("memoryManagement");

    // Move the layer requested to be reallocated so that it starts just after
    // the newest allocated layer
    float* newLayerStartAddress = makeRoom(newSizeInWords);
    layers[layerIndex]->moveDeviceData(newLayerStartAddress);

    Profiler::pop();
}

float* MemoryManager::makeRoom(size_t requiredSpaceInWords) {

    Profiler::push("memoryMakeRoom");

    // When doing the forward pass, start at the lowest allocated layer and keep
    // spilling layers until we have enough free space to allocate the requested
    // amount of words. Do the reverse on the backwards pass

    // After we've made room, take the specified layer and move it so that it
    // starts just after the newest allocated layer

    if(direction == trainingDirection::forwards) {
        size_t freeWords = 0;

        DNNLayer* oldestLayer;
        DNNLayer* newestLayer;

        do {
            oldestLayer = getOldestAllocatedLayer();
            newestLayer = getNewestAllocatedLayer();

            // In scenarios with very low memory, all layers will be spilled. In
            // this case we can allocate a new layer at the start of the memory
            // pool
            if(oldestLayer == NULL && newestLayer == NULL) {
                Profiler::pop();
                return pool;
            }

            // First, check if we already have enough space
            // Trivial case; no layer looping, all layers are allocated from the
            // lowest to highest memory indices
            size_t wordsFromNewestToLimit = 0;
            if(newestLayer->startAddress > oldestLayer->startAddress) {
                wordsFromNewestToLimit = oldestLayer->startAddress - this->pool;
            }

            // Non-trivial case, newest allocated layer in the transformer is
            // present below the lowest allocated layer
            size_t wordsBetweenNewestAndOldest = 0;
            float* newestLayerEndAddress = newestLayer->startAddress + newestLayer->getLayerSizeInWords();
            if(oldestLayer->startAddress > newestLayerEndAddress) {
                wordsBetweenNewestAndOldest = oldestLayer->startAddress - newestLayerEndAddress;
            }

            freeWords = max(wordsFromNewestToLimit, wordsBetweenNewestAndOldest);
                
            if(freeWords < requiredSpaceInWords) {
                oldestLayer->spill();
            }

        } while(freeWords < requiredSpaceInWords);
        
        // Find a suitable pointer for the requested space
        float* newestAllocatedLayerEndAddress = newestLayer->startAddress + newestLayer->getLayerSizeInWords();

        int newestToOldest = oldestLayer->startAddress - newestAllocatedLayerEndAddress;
        int newestToUpperLimit = (pool + poolSizeInWords) - newestAllocatedLayerEndAddress;
        int startToOldest = oldestLayer->startAddress - pool;

        float* newLayerStartAddress = NULL;

        // First, try moving it to just after the newest layer
        if(newestToUpperLimit >= requiredSpaceInWords && newestToOldest >= requiredSpaceInWords) {
            newLayerStartAddress = newestAllocatedLayerEndAddress;
        
        // Sometimes there may not be enough memory above the newest layer and
        // we'll need to wrap around to the beginning
        } else if(startToOldest >= requiredSpaceInWords) {
            newLayerStartAddress = pool;
        } else {
            spdlog::error("Couldn't find a suitable place to reallocate a layer's data");
        }
        
        Profiler::pop();
        return newLayerStartAddress;

    } else if(direction == trainingDirection::backwards) {
        size_t freeWords = 0;

        DNNLayer* oldestLayer;
        DNNLayer* newestLayer;

        do {
            oldestLayer = getOldestAllocatedLayer();
            newestLayer = getNewestAllocatedLayer();

            // In scenarios with very low memory, all layers will be spilled. In
            // this case we can allocate a new layer just below the top of the
            // memory pool
            if(oldestLayer == NULL && newestLayer == NULL) {
                Profiler::pop();
                return (pool + poolSizeInWords) - requiredSpaceInWords;
            }

            // First, check if we already have enough space
            // Trivial case; no layer looping, all layers are allocated from the
            // lowest to highest memory indices
            size_t wordsFromOldestToPoolLimit = 0;
            if(newestLayer->startAddress < oldestLayer->startAddress) {
                wordsFromOldestToPoolLimit = (pool + poolSizeInWords) - (oldestLayer->startAddress + oldestLayer->getLayerSizeInWords());
            }

            // Non-trivial case, newest allocated layer in the transformer is
            // present below the lowest allocated layer
            size_t wordsBetweenNewestAndOldest = 0;
            float* oldestLayerEndAddress = oldestLayer->startAddress + oldestLayer->getLayerSizeInWords();
            if(newestLayer->startAddress > oldestLayerEndAddress) {
                wordsBetweenNewestAndOldest = newestLayer->startAddress - oldestLayerEndAddress;
            }

            freeWords = max(wordsFromOldestToPoolLimit, wordsBetweenNewestAndOldest);
                
            if(freeWords < requiredSpaceInWords) {
                oldestLayer->spill();
            }

        } while(freeWords < requiredSpaceInWords);
        
        // Find a suitable pointer for the requested space
        float* oldestAllocatedLayerEndAddress = oldestLayer->startAddress + oldestLayer->getLayerSizeInWords();

        int oldestToNewest = newestLayer->startAddress - oldestAllocatedLayerEndAddress;
        int newestToPoolStart = newestLayer->startAddress - pool;
        int oldestToUpperLimit = (pool + poolSizeInWords) - (oldestLayer->startAddress + oldestLayer->getLayerSizeInWords());

        float* newLayerStartAddress = NULL;

        // First, try just before the newest layer
        if(newestToPoolStart >= requiredSpaceInWords) {
            newLayerStartAddress = newestLayer->startAddress - requiredSpaceInWords;
        
        // Sometimes there may not be enough memory below the newest layer and
        // we'll need to wrap around to the top
        } else if(oldestToUpperLimit >= requiredSpaceInWords) {
            newLayerStartAddress = (pool + poolSizeInWords) - requiredSpaceInWords;
        } else {
            spdlog::error("Couldn't find a suitable place to reallocate a layer's data");
        }
        
        Profiler::pop();
        return newLayerStartAddress;
    }

    // This should never be reached
    spdlog::error("Reached the end of {} at {}:{} without providing a memory address", __func__, __FILE__, __LINE__);
    Profiler::pop();
    return NULL;
}

void MemoryManager::changeDirections() {

    Profiler::push("memoryManagement");

    if(direction == trainingDirection::forwards) {
        direction = trainingDirection::backwards;
    } else if(direction == trainingDirection::backwards) {
        direction = trainingDirection::forwards;
    }

    // // Invert all the ages (newest age becomes oldest age and vice versa)
    // int oldestAge = 0;
    // for(int i = 0; i < layers.size(); i++) {
    //     if(!layers[i]->spilled && layers[i]->age > oldestAge) {
    //         oldestAge = layers[i]->age;
    //     }
    // }

    // for(int i = 0; i < layers.size(); i++) {
    //     if(!layers[i]->spilled) {
    //         layers[i]->age = oldestAge - layers[i]->age;
    //     }
    // }

    spdlog::info("Memory manager training direction changed");
    Profiler::pop();
    reportUsage();
}

void MemoryManager::reportUsage() {

    Profiler::push("messages");

#ifdef PRINT_INFO
    spdlog::info("BERT Training memory usage:");
    printf("Memory manager start address: %p\n", this->pool);
    
    size_t totalUsage = 0;
    size_t totalLiveUsage = 0;
    for(int i = 0; i < this->layers.size(); i++) {
        size_t layerUsage = this->layers[i]->getLayerSizeInWords();
        printf("Layer %d start: +%.3f MB (size: %.3f MB, age: %d)", i, ((this->layers[i]->startAddress) - pool) * 4e-6, layerUsage * 4e-6, layers[i]->age);

        totalUsage += layerUsage;
        if(!this->layers[i]->spilled) {
            totalLiveUsage += layerUsage;
        } else {
            printf(" (spilled)");
        }

        size_t previousLayerIndex = (i - 1) % layers.size();
        if(layers[previousLayerIndex]->startAddress + layers[previousLayerIndex]->getLayerSizeInWords() > this->layers[i]->startAddress
            && layers[i]->startAddress > layers[previousLayerIndex]->startAddress
            && !layers[previousLayerIndex]->spilled) {
            printf(" (overlapped by previous layer!)");
        }

        printf("\n");
    }

    totalUsage *= 4;
    totalLiveUsage *= 4;
    DNNLayer* oldestLayer = getOldestAllocatedLayer();
    DNNLayer* newestLayer = getNewestAllocatedLayer();
    if(oldestLayer != NULL && newestLayer != NULL) {
        printf("Oldest allocated layer: %d, newest allocated layer: %d\n", oldestLayer->index, newestLayer->index);
    }
    printf("Total memory manager usage: %.3f MB (%.3f MB live)\n", totalUsage / 1e6, totalLiveUsage / 1e6);
#endif

    Profiler::pop();
}

DNNLayer* MemoryManager::getNextActiveLayer() {

    Profiler::push("memoryManagement");

    float* layerStartAddress = pool;
    
    for(int i = 0; i < layers.size(); i++) {
        layerStartAddress = layers[i]->startAddress + layers[i]->getLayerSizeInWords();
    }

    ageLayers();
    DNNLayer* newLayer = new DNNLayer(layers.size(), layerStartAddress, pool + poolSizeInWords, this);
    this->layers.push_back(newLayer);

    Profiler::pop();
    return newLayer;
}

DNNLayer* MemoryManager::getNewestAllocatedLayer() {

    Profiler::push("memoryManagement");

    int newestLayerIndex = -1;
    int ageOfNewestLayer = INT_MAX;
    
    // It is expected that each live layer has a unique age value
    // The newest allocated layer has the lowest age value (should be 0)
    for(int i = 0; i < layers.size(); i++) {
        if(layers[i]->spilled == false && layers[i]->age < ageOfNewestLayer) {
            newestLayerIndex = i;
            ageOfNewestLayer = layers[i]->age;
        }
    }

    Profiler::pop();
    
    if(newestLayerIndex == -1) {
        return NULL;
    } else {
        return layers[newestLayerIndex];
    }
}

DNNLayer* MemoryManager::getOldestAllocatedLayer() {

    Profiler::push("memoryManagement");

    int oldestLayerIndex = -1;
    int ageOfOldestLayer = INT_MIN;
    
    // It is expected that each live layer has a unique age value
    // The oldest allocated layer has the largest age value
    for(int i = 0; i < layers.size(); i++) {
        if(layers[i]->spilled == false && layers[i]->age > ageOfOldestLayer) {
            oldestLayerIndex = i;
            ageOfOldestLayer = layers[i]->age;
        }
    }

    Profiler::pop();

    if(oldestLayerIndex == -1) {
        return NULL;
    } else {
        return layers[oldestLayerIndex];
    }
}

void MemoryManager::ageLayers() {

    if(!enableOptimizations)
        return;
    Profiler::push("memoryManagement");

    for(int i = 0; i < layers.size(); i++) {
        if(layers[i]->spilled == false) {
            layers[i]->age++;
        }
    }

    Profiler::pop();
}