#pragma once

#include <pablo/pablo_kernel.h>

namespace kernel {
    
//
//  Given a set of insertion amounts (numbers of zeroes to be inserted
//  at indexed positions, and a stream set (possibly multiplexed) identifying
//  position indices at which insertion is to occur, a bixNum stream set is
//  calculated such that the bixNum at position p is n if the zeroes to be
//  inserted at the position is n, or 0 if no insertion is to occur.
//  The insertMarks stream set may have one stream each for the element
//  of the insertion amounts vector, or may be multiplexed.   If multiplexed,
//  the insertMarks streamset is a BixNum encoding the 1-based index of
//  of the insertionAmts vector, with a BixNum value of 0 indicating that no
//  insertion is to occur at the identified position.
//
//  The result may then be used for calculation of a SpreadMask by InsertionSpreadMask.
//

class ZeroInsertBixNum final : public pablo::PabloKernel {
public:
    ZeroInsertBixNum(LLVMTypeSystemInterface & ts, const std::vector<unsigned> &insertAmts,
                       StreamSet * insertMarks, StreamSet * insertBixNum);
    void generatePabloMethod() override;
    bool hasSignature() const override { return true; }
    llvm::StringRef getSignature() const override {
        return mSignature;
    }
private:
    const std::vector<unsigned>     mInsertAmounts;
    const bool                      mMultiplexing;
    const unsigned                  mBixNumBits;
    const std::string               mSignature;
};

class StringReplaceKernel final : public pablo::PabloKernel {
public:
    StringReplaceKernel(LLVMTypeSystemInterface & ts, const std::vector<std::string> & insertStrs,
                        StreamSet * basis, StreamSet * spreadMask,
                        StreamSet * insertMarks, StreamSet * runIndex,
                        StreamSet * output,
                        int offset = 0);
    void generatePabloMethod() override;
    bool hasSignature() const override { return true; }
    llvm::StringRef getSignature() const override {
        return mSignature;
    }
private:
    StringReplaceKernel(LLVMTypeSystemInterface & ts, std::string && signature,
                        const std::vector<std::string> & insertStrs,
                        StreamSet * basis, StreamSet * spreadMask,
                        StreamSet * insertMarks, StreamSet * runIndex,
                        StreamSet * output,
                        int offset = 0);
private:
    const std::vector<std::string>  mInsertStrings;
    const bool                      mMultiplexing;
    const int                       mMarkOffset;
    const std::string               mSignature;
};
}

