/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <kernel/bitwise/bixlogic.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/pablo_kernel.h>

using namespace pablo;

namespace kernel {

class InversionKernel : public pablo::PabloKernel {
public:
    InversionKernel(LLVMTypeSystemInterface & ts, StreamSet * mask, StreamSet * inverted);
protected:
    void generatePabloMethod() override;
};

InversionKernel::InversionKernel(LLVMTypeSystemInterface & ts,
                                 StreamSet * mask, StreamSet * inverted)
: PabloKernel(ts, "Invert",
              {Binding{"mask", mask}},
              {Binding{"inverted", inverted}}) {}

void InversionKernel::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * mask = getInputStreamSet("mask")[0];
    PabloAST * inverted = pb.createInFile(pb.createNot(mask));
    Var * outVar = getOutputStreamVar("inverted");
    pb.createAssign(pb.createExtract(outVar, pb.getInteger(0)), inverted);
}

void Invert(PipelineBuilder & P, StreamSet * mask, StreamSet * inverted) {
    P.CreateKernelCall<InversionKernel>(mask, inverted);
}

enum class BitwiseOp {Or, Xor, And};

std::string uniqueOperationName(BitwiseOp op, CombiningKind k,
                                StreamSet * source, StreamSet * toCombine) {
    std::string tmp;
    llvm::raw_string_ostream out(tmp);
    if (op == BitwiseOp::Or) out << "OrCombine";
    else if (op == BitwiseOp::Xor) out << "XorCombine";
    else out << "AndCombine";
    out << "_" << source->shapeString();
    out << "_" << toCombine->shapeString();
    if (k == CombiningKind::InOut) {
        out << "+InOut";
    }
    return out.str();
}

class BitwiseCombine : public PabloKernel {
public:
    BitwiseCombine(LLVMTypeSystemInterface & ts,
                   BitwiseOp op,
                   StreamSet * source, StreamSet * toCombine,
                   StreamSet * combined, CombiningKind k)
    : PabloKernel(ts, uniqueOperationName(op, k, source, toCombine),
                  {Binding{"source", source}, Binding{"toCombine", toCombine}},
                  {}), mOp(op)
    {
        if (k == CombiningKind::InOut) {
            mOutputStreamSets.push_back(Binding{"combined", combined, FixedRate(), InOut("source")});
        } else {
            mOutputStreamSets.push_back(Binding{"combined", combined});
        }
    }
protected:
    void generatePabloMethod() override;
private:
    BitwiseOp mOp;
};

void BitwiseCombine::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> source = getInputStreamSet("source");
    std::vector<PabloAST *> toCombine = getInputStreamSet("toCombine");
    std::vector<PabloAST *> combined(source.size());
    for (unsigned i = 0; i < toCombine.size(); i++) {
        if (mOp == BitwiseOp::Or) {
            combined[i] = pb.createOr(source[i], toCombine[i]);
        } else if (mOp == BitwiseOp::Xor) {
            combined[i] = pb.createXor(source[i], toCombine[i]);
        } else {
            combined[i] = pb.createAnd(source[i], toCombine[i]);
        }
    }
    for (unsigned i = toCombine.size(); i < source.size(); i++) {
        combined[i] = source[i];
    }
    writeOutputStreamSet("combined", combined);
}

void OrCombine(PipelineBuilder & P,
               StreamSet * source, StreamSet * toCombine,
               StreamSet * combined, CombiningKind k) {
    P.CreateKernelCall<BitwiseCombine>(BitwiseOp::Or, source, toCombine, combined, k);
}

void XorCombine(PipelineBuilder & P,
                StreamSet * source, StreamSet * toCombine,
                StreamSet * combined, CombiningKind k) {
    P.CreateKernelCall<BitwiseCombine>(BitwiseOp::Xor, source, toCombine, combined, k);
}

void AndCombine(PipelineBuilder & P,
                StreamSet * source, StreamSet * toCombine,
                StreamSet * combined, CombiningKind k) {
    P.CreateKernelCall<BitwiseCombine>(BitwiseOp::And, source, toCombine, combined, k);
}

std::string ZBM_unique_name(CombiningKind k, StreamSet * source) {
    std::string tmp;
    llvm::raw_string_ostream out(tmp);
    out << "ZeroByMask_";
    out << source->shapeString();
    if (k == CombiningKind::InOut) {
        out << "+InOut";
    }
    return out.str();
}


class ZeroByMaskKernel : public PabloKernel {
public:
    ZeroByMaskKernel(LLVMTypeSystemInterface & ts,
                     StreamSet * mask, StreamSet * source,
                     StreamSet * masked, CombiningKind k)
    : PabloKernel(ts, ZBM_unique_name(k, source),
                  {Binding{"mask", mask}, Binding{"source", source}},
                  {})
    {
        if (k == CombiningKind::InOut) {
            mOutputStreamSets.push_back(Binding{"masked", masked, FixedRate(), InOut("source")});
        } else {
            mOutputStreamSets.push_back(Binding{"masked", masked});
        }
    }
protected:
    void generatePabloMethod() override;
};

void ZeroByMaskKernel::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * mask = getInputStreamSet("mask")[0];
    std::vector<PabloAST *> source = getInputStreamSet("source");
    std::vector<PabloAST *> masked(source.size());
    for (unsigned i = 0; i < source.size(); i++) {
        masked[i] = pb.createAnd(mask, source[i]);
    }
    writeOutputStreamSet("masked", masked);
}

void ZeroByMask(PipelineBuilder & P,
                StreamSet * mask, StreamSet * source,
                StreamSet * masked, CombiningKind k) {
    P.CreateKernelCall<ZeroByMaskKernel>(mask, source, masked, k);
}

}
