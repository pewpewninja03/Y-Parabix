/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <kernel/core/relationship.h>
#include <kernel/core/kernel.h>

using namespace kernel;
namespace bixnum {

// Helpers to generate unique names for operations depending on
// stream set shapes and unsigned integer constatns.
std::string uniqueBinaryOperationName(std::string, StreamSet * a, StreamSet * b, StreamSet * rslt);
std::string uniqueBinaryImmediateName(std::string, StreamSet * a, unsigned b, StreamSet * rslt);

//
//  Binary modular arithmetic operations on BixNums.
//
//  The number of significant bits is determined by the result BixNum.
//  Input BixNums are truncated or zero-extended to match.
//
//  Example use:
//  StreamSet * rslt = P.CreateStreamSet(significant_bits);
//  P.CreateKernelCall<bixnum::Add>(op1, op2, rslt);

class Add final : public pablo::PabloKernel {
public:
    Add(LLVMTypeSystemInterface & ts, StreamSet * a, StreamSet * b, StreamSet * sum)
    : pablo::PabloKernel(ts, uniqueBinaryOperationName("Add", a, b, sum),
                         {Binding{"a", a}, Binding{"b", b}}, {Binding{"sum", sum}}),
    mBixBits(sum->getNumElements()) {
    }
protected:
    void generatePabloMethod() override;
private:
    const unsigned mBixBits;
};

class Add_immediate final : public pablo::PabloKernel {
public:
    Add_immediate(LLVMTypeSystemInterface & ts, StreamSet * a, unsigned b, StreamSet * sum)
    : pablo::PabloKernel(ts, uniqueBinaryImmediateName("Addi", a, b, sum),
                         {Binding{"a", a}}, {Binding{"sum", sum}}),
    mBixBits(sum->getNumElements()), mImmediate(b) {
    }
protected:
    void generatePabloMethod() override;
private:
    const unsigned mBixBits;
    const unsigned mImmediate;
};

class Sub final : public pablo::PabloKernel {
public:
    Sub(LLVMTypeSystemInterface & ts, StreamSet * a, StreamSet * b, StreamSet * diff)
    : pablo::PabloKernel(ts, uniqueBinaryOperationName("Sub", a, b, diff),
                         {Binding{"a", a}, Binding{"b", b}}, {Binding{"diff", diff}}),
    mBixBits(diff->getNumElements()) {
    }
protected:
    void generatePabloMethod() override;
private:
    const unsigned mBixBits;
};

class Sub_immediate final : public pablo::PabloKernel {
public:
    Sub_immediate(LLVMTypeSystemInterface & ts, StreamSet * a, unsigned b, StreamSet * diff)
    : pablo::PabloKernel(ts, uniqueBinaryImmediateName("Subi", a, b, diff),
                         {Binding{"a", a}}, {Binding{"diff", diff}}),
    mBixBits(diff->getNumElements()), mImmediate(b) {
    }
protected:
    void generatePabloMethod() override;
private:
    const unsigned mBixBits;
    const unsigned mImmediate;
};

class Mul_immediate final : public pablo::PabloKernel {
public:
    Mul_immediate(LLVMTypeSystemInterface & ts, StreamSet * a, unsigned b, StreamSet * product)
    : pablo::PabloKernel(ts, uniqueBinaryImmediateName("Muli", a, b, product),
                         {Binding{"a", a}}, {Binding{"product", product}}),
    mBixBits(product->getNumElements()), mImmediate(b) {
    }
protected:
    void generatePabloMethod() override;
private:
    const unsigned mBixBits;
    const unsigned mImmediate;
};

class EQ_immediate final : public pablo::PabloKernel {
public:
    EQ_immediate(LLVMTypeSystemInterface & ts, StreamSet * a, unsigned b, StreamSet * rslt)
    : pablo::PabloKernel(ts, uniqueBinaryImmediateName("EQi", a, b, rslt),
                         {Binding{"a", a}}, {Binding{"rslt", rslt}}),
    mImmediate(b) {
    }
protected:
    void generatePabloMethod() override;
private:
    const unsigned mImmediate;
};

class NEQ_immediate final : public pablo::PabloKernel {
public:
    NEQ_immediate(LLVMTypeSystemInterface & ts, StreamSet * a, unsigned b, StreamSet * rslt)
    : pablo::PabloKernel(ts, uniqueBinaryImmediateName("NEQi", a, b, rslt),
                         {Binding{"a", a}}, {Binding{"rslt", rslt}}),
    mImmediate(b) {
    }
protected:
    void generatePabloMethod() override;
private:
    const unsigned mImmediate;
};

class UGT_immediate final : public pablo::PabloKernel {
public:
    UGT_immediate(LLVMTypeSystemInterface & ts, StreamSet * a, unsigned b, StreamSet * rslt)
    : pablo::PabloKernel(ts, uniqueBinaryImmediateName("UGTi", a, b, rslt),
                         {Binding{"a", a}}, {Binding{"rslt", rslt}}),
    mImmediate(b) {
    }
protected:
    void generatePabloMethod() override;
private:
    const unsigned mImmediate;
};

class UGE_immediate final : public pablo::PabloKernel {
public:
    UGE_immediate(LLVMTypeSystemInterface & ts, StreamSet * a, unsigned b, StreamSet * rslt)
    : pablo::PabloKernel(ts, uniqueBinaryImmediateName("UGEi", a, b, rslt),
                         {Binding{"a", a}}, {Binding{"rslt", rslt}}),
    mImmediate(b) {
    }
protected:
    void generatePabloMethod() override;
private:
    const unsigned mImmediate;
};

class ULT_immediate final : public pablo::PabloKernel {
public:
    ULT_immediate(LLVMTypeSystemInterface & ts, StreamSet * a, unsigned b, StreamSet * rslt)
    : pablo::PabloKernel(ts, uniqueBinaryImmediateName("ULT", a, b, rslt),
                         {Binding{"a", a}}, {Binding{"rslt", rslt}}),
    mImmediate(b) {
    }
protected:
    void generatePabloMethod() override;
private:
    const unsigned mImmediate;
};

class ULE_immediate final : public pablo::PabloKernel {
public:
    ULE_immediate(LLVMTypeSystemInterface & ts, StreamSet * a, unsigned b, StreamSet * rslt)
    : pablo::PabloKernel(ts, uniqueBinaryImmediateName("ULEi", a, b, rslt),
                         {Binding{"a", a}}, {Binding{"rslt", rslt}}),
    mImmediate(b) {
    }
protected:
    void generatePabloMethod() override;
private:
    const unsigned mImmediate;
};

}
