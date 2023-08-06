#include "../include/profiler.hpp"

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

using namespace std;

map<string, double> Profiler::categoryTimes;
vector<string> Profiler::timerStack;
chrono::time_point<chrono::high_resolution_clock> Profiler::timerStart;

void Profiler::setup() {
    categoryTimes = {
        {"cudaSetup", 0},
        {"cudnnExecution", 0},
        {"memoryLoadRandom", 0},
        {"memoryMakeRoom", 0},
        {"memoryManagement", 0},
        {"memoryMoveDeviceData", 0},
        {"memoryReadFromDevice", 0},
        {"memorySpill", 0},
        {"memoryUnspill", 0},
        {"messages", 0},
        {"otherTraining", 0},
        {"randomNumberGeneration", 0},
    };

    timerStack = vector<string>();
    timerStack.push_back("otherTraining");
    timerStart = chrono::high_resolution_clock::now();
}

void Profiler::push(string timerName) {
    // When pushing someting to the stack of timers:
    // Pause the previous timer, add the value to the total
    // Append the new timer to the list
    // Start the new timer

    if(categoryTimes.contains(timerName)) {

        chrono::time_point<chrono::high_resolution_clock> curr_time = chrono::high_resolution_clock::now();
        double elapsedTime = chrono::duration_cast<chrono::milliseconds>(curr_time - timerStart).count() * 0.001;

        string currentTimer = timerStack[timerStack.size() - 1];
        categoryTimes[currentTimer] = categoryTimes.find(currentTimer)->second + elapsedTime;

        timerStart = curr_time;

        timerStack.push_back(timerName);
    } else {
        spdlog::error("Tried starting an unknown timer: {}", timerName);
        exit(1);
    }
}

void Profiler::pop() {
    // Stop the previous timer
    // Add it to the category
    // Pop the latest category from the stack

    chrono::time_point<chrono::high_resolution_clock> curr_time = chrono::high_resolution_clock::now();
    double elapsedTime = chrono::duration_cast<chrono::milliseconds>(curr_time - timerStart).count() * 0.001;

    string currentTimer = timerStack[timerStack.size() - 1];
    categoryTimes[currentTimer] = categoryTimes.find(currentTimer)->second + elapsedTime;

    timerStart = curr_time;

    timerStack.pop_back();
}

void Profiler::finish() {
    chrono::time_point<chrono::high_resolution_clock> curr_time = chrono::high_resolution_clock::now();
    double elapsedTime = chrono::duration_cast<chrono::milliseconds>(curr_time - timerStart).count() * 0.001;

    string currentTimer = timerStack[timerStack.size() - 1];
    categoryTimes[currentTimer] = categoryTimes.find(currentTimer)->second + elapsedTime;
}

void Profiler::report() {
    spdlog::info("Profiler report:");

    for(map<string, double>::iterator timerCategoryIterator = categoryTimes.begin();
        timerCategoryIterator != categoryTimes.end();
        timerCategoryIterator++) {
        
        printf("\t%s: %.3fs\n", timerCategoryIterator->first.c_str(), timerCategoryIterator->second);
    }
}