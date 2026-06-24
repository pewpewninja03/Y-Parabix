/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <kernel/bitwise/bixlogic.h>
#include <kernel/core/kernel_builder.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/pablo_kernel.h>

using namespace llvm;
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

class StreamsCombine : public BlockOrientedKernel {
public:
    StreamsCombine(LLVMTypeSystemInterface & ts,
                   BitwiseOp op,
                   StreamSet * source, StreamSet * toCombine,
                   StreamSet * combined, CombiningKind k);

protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
    Value * combine(KernelBuilder & b, Value * x, Value * y);
private:
    BitwiseOp mOp;
    CombiningKind mKind;
};

Value * StreamsCombine::combine(KernelBuilder & b, Value * x, Value * y) {
    if (mOp == BitwiseOp::Or) {
        return b.CreateOr(x, y);
    } else if (mOp == BitwiseOp::Xor) {
        return b.CreateXor(x, y);
    } else {
        return b.CreateAnd(x, y);
    }
}

StreamsCombine::StreamsCombine(LLVMTypeSystemInterface & ts,
                               BitwiseOp op,
                               StreamSet * source, StreamSet * toCombine,
                               StreamSet * combined, CombiningKind k)
: BlockOrientedKernel(ts, uniqueOperationName(op, k, source, toCombine), 
    {Binding{"source", source}, Binding{"toCombine", toCombine}}, {}, {}, {}, {}), mOp(op), mKind(k) {
        if (k == CombiningKind::InOut) {
            mOutputStreamSets.push_back(Binding{"combined", combined, FixedRate(), InOut("source")});
        } else {
            mOutputStreamSets.push_back(Binding{"combined", combined});
        }
        assert(source->getFieldWidth() == toCombine->getFieldWidth());
        assert(source->getFieldWidth() == combined->getFieldWidth());
        assert(source->getNumElements() == combined->getNumElements());
        assert(source->getNumElements() >= toCombine->getNumElements());
}

void StreamsCombine::generateDoBlockMethod(KernelBuilder & b) {
    const auto fw = getInputStreamSet(0)->getFieldWidth();
    const auto n = getInputStreamSet(0)->getNumElements();
    const auto m = getInputStreamSet(1)->getNumElements();
    for (unsigned i = 0; i < m; ++i) {
        if (fw == 1) {
            Value * src = b.loadInputStreamBlock("source", b.getInt32(i));
            Value * toCombine = b.loadInputStreamBlock("toCombine", b.getInt32(i));
            b.storeOutputStreamBlock("combined", b.getInt32(i), combine(b, src, toCombine));
        } else {
            for (unsigned k = 0; k < fw; k++) {
                Value * pack1 = b.loadInputStreamPack("source", b.getInt32(i), b.getInt32(k));
                Value * pack2 = b.loadInputStreamPack("toCombine", b.getInt32(i), b.getInt32(k));
                b.storeOutputStreamPack("combined", b.getInt32(i), b.getInt32(k), combine(b, pack1, pack2));
            }
        }
    }
    if (mKind != CombiningKind::InOut) {
        for (unsigned i = m; i < n; ++i) {
            if (fw == 1) {
                Value * src = b.loadInputStreamBlock("source", b.getInt32(i));
                b.storeOutputStreamBlock("combined", b.getInt32(i), src);
            } else {
                for (unsigned k = 0; k < fw; k++) {
                    Value * pack = b.loadInputStreamPack("source", b.getInt32(i), b.getInt32(k));
                    b.storeOutputStreamPack("combined", b.getInt32(i), b.getInt32(k), pack);
                }
            }
        }
    }
}

void OrCombine(PipelineBuilder & P,
               StreamSet * source, StreamSet * toCombine,
               StreamSet * combined, CombiningKind k) {
    P.CreateKernelCall<StreamsCombine>(BitwiseOp::Or, source, toCombine, combined, k);
}

void XorCombine(PipelineBuilder & P,
                StreamSet * source, StreamSet * toCombine,
                StreamSet * combined, CombiningKind k) {
    P.CreateKernelCall<StreamsCombine>(BitwiseOp::Xor, source, toCombine, combined, k);
}

void AndCombine(PipelineBuilder & P,
                StreamSet * source, StreamSet * toCombine,
                StreamSet * combined, CombiningKind k) {
    P.CreateKernelCall<StreamsCombine>(BitwiseOp::And, source, toCombine, combined, k);
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
