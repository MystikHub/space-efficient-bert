#include <cstring>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "include/bert.hpp"
#include "include/profiler.hpp"
#include "include/token_reader.hpp"

using namespace std;

int main(int argc, char** argv) {

    // spdlog::set_level(spdlog::level::off);
#define PRINT_INFO

    int memoryPoolSize = -1;
    bool parsingOk = false;

    if(argc >= 5 && strcmp(argv[3], "--optimizations") == 0) {
        
        if(strcmp(argv[4], "on") == 0) {

            Profiler::setup();

            memoryPoolSize = stoi(string(argv[5]));
            BERT bert = BERT(memoryPoolSize * 1e6, true);

            TokenReader trainingData = TokenReader();
            int nEpochs = stoi(string(argv[2]));
            bert.train(trainingData, nEpochs);

            parsingOk = true;

            Profiler::finish();
            Profiler::report();

        } else if(strcmp(argv[4], "off") == 0) {

            Profiler::setup();

            BERT bert = BERT(memoryPoolSize * 1e6, false);

            TokenReader trainingData = TokenReader();
            int nEpochs = stoi(string(argv[2]));
            bert.train(trainingData, nEpochs);

            parsingOk = true;

            Profiler::finish();
            Profiler::report();
        }
    }
    
    if(!parsingOk) {
        spdlog::error("Usage:\n");
        spdlog::error("{} --epochs <int> --optimizations <on/off> <memory pool size (in MB)>\n", argv[0]);
    }

    return 0;
}