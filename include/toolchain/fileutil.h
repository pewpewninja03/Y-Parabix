/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 *
 */
#pragma once
 
#include <string>
#include <util/aligned_allocator.h>

extern bool NoOSFileCaching;

bool canMMap(const std::string & fileName);

class AlignedFileBuffer {
public:
    AlignedFileBuffer();
    void load(const std::string & fileName, bool preferMMap = false);
    size_t getBufSize();
    char * getBuf();
    void release();
private:
    static const unsigned mAlignment = 4096;
    AlignedAllocator<char, mAlignment> mAlloc;
    std::string mFileName;
    bool mDidMMap;
    char * mBuffer;
    size_t mAlignedSize;
    size_t mBufSize;
};
