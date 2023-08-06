#ifndef TOKEN_READER_HPP
#define TOKEN_READER_HPP

#include <fstream>
#include <string>
#include <vector>

using namespace std;

class TokenReader {
public:
    TokenReader();
    TokenReader(string tokenFile);
    TokenReader(const TokenReader& source);
    ~TokenReader();

    string getFilePath();
    bool reachedEnd();
    void reset();

    TokenReader operator++(int);
    vector<unsigned int> operator*();

private:
    string filepath;
    ifstream inputFileStream;
    streampos location;
    bool readable;

    string lastReadLine;
    vector<unsigned int> lastReadTokenIds;

    void readAndParseTokenIds();
};

#endif