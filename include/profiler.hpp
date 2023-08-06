#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <chrono>
#include <map>
#include <string>
#include <vector>

using namespace std;

class Profiler {
public:
    static map<string, double> categoryTimes;

    static vector<string> timerStack;
    static chrono::time_point<chrono::high_resolution_clock> timerStart;
    static void setup();
    static void push(string timerName);
    static void pop();
    static void finish();
    static void report();
};

#endif