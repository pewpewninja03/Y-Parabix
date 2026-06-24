#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel

namespace kernel {

// Extend input bit streams by one position with adding a 1 bit.
class AddSentinel final : public pablo::PabloKernel {
public:
    AddSentinel(LLVMTypeSystemInterface & ts,
               StreamSet * const input, StreamSet * const output);
    void generatePabloMethod() override;
};

// Return a stream have a single 1 bit immediately after 
// the last bit of the input stream.
class EOFbit final : public pablo::PabloKernel {
public:
    EOFbit(LLVMTypeSystemInterface & ts,
           StreamSet * const input, StreamSet * const output);
    void generatePabloMethod() override;
};

}
