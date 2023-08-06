#include "../include/token_reader.hpp"

#include <fstream>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

TokenReader::TokenReader() {
    filepath = "";
    location = 0;
    lastReadLine = "";
    readable = false;
    lastReadTokenIds = vector<unsigned int>();
}

TokenReader::TokenReader(string dataFilePath)
    : filepath(dataFilePath),
    location(0),
    lastReadLine(""),
    readable(true) {

    inputFileStream.open(filepath, ifstream::in);

    // Read the first line
    readAndParseTokenIds();
}

TokenReader::TokenReader(const TokenReader& source)
    : filepath(source.filepath),
    lastReadLine(source.lastReadLine),
    lastReadTokenIds(source.lastReadTokenIds),
    readable(source.readable) {

    // Can't create an input stream using another input stream
    inputFileStream.open(filepath, ifstream::in);
    inputFileStream.seekg(source.location);
}

TokenReader::~TokenReader() {
    if(inputFileStream.is_open())
        inputFileStream.close();
}

bool TokenReader::reachedEnd() {
    if(readable)
        return inputFileStream.eof();
    else
        return location;
}

TokenReader TokenReader::operator++(int) {

    if(readable) {
        // Read the next line
        readAndParseTokenIds();
        
        // Update the information required by the copy constructor
        location = inputFileStream.tellg();
    } else {
        location = 1;
    }

    return *this;
}

vector<unsigned int> TokenReader::operator*() {
    return lastReadTokenIds;
}

string TokenReader::getFilePath() {
    return filepath;
}

void TokenReader::readAndParseTokenIds() {

    if(!readable)
        return;

    getline(inputFileStream, lastReadLine);

    string consumerBuffer = string(lastReadLine);
    size_t delimiterLocation = 0;

    lastReadTokenIds = vector<unsigned int>();

    while((delimiterLocation = consumerBuffer.find(' ')) != string::npos) {
        string tokenIdString = consumerBuffer.substr(0, delimiterLocation);
        unsigned int newTokenId = (unsigned int) stoi(tokenIdString);
        consumerBuffer.erase(0, delimiterLocation + 1);

        lastReadTokenIds.push_back(newTokenId);
    }

    // Last token (not followed by a delimiter)
    string tokenIdString = consumerBuffer.substr(0, delimiterLocation);
    unsigned int newTokenId = (unsigned int) stoi(tokenIdString);
    consumerBuffer.erase(0, delimiterLocation + 1);

    lastReadTokenIds.push_back(newTokenId);

    spdlog::info("Read {} tokens from {}", lastReadTokenIds.size(), filepath);
}

void TokenReader::reset() {
    if(readable) {
        if(inputFileStream.is_open())
            inputFileStream.close();
        
        inputFileStream.open(filepath, ifstream::in);
        location = inputFileStream.tellg();

    } else {
        location = 0;
    }
}