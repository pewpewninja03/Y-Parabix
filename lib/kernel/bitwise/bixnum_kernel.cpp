/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <kernel/bitwise/bixnum_kernel.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>

using namespace kernel;
namespace bixnum {

std::string uniqueBinaryOperationName(std::string op, StreamSet * a, StreamSet * b, StreamSet * rslt) {
    std::string tmp;
    llvm::raw_string_ostream out(tmp);
    out << op << "_" << a->shapeString();
    out << "_" << b->shapeString();
    out << ":" << rslt->shapeString();
    return out.str();
}

std::string uniqueBinaryImmediateName(std::string op, StreamSet * a, unsigned b, StreamSet * rslt) {
    std::string tmp;
    llvm::raw_string_ostream out(tmp);
    out << op << "_" << a->shapeString();
    out << "_imm<" << std::to_string(b) << ">";
    out << ":" << rslt->shapeString();
    return out.str();
}

void Add::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::BixNum a = getInputStreamSet("a");
    pablo::BixNum b = getInputStreamSet("b");
    pablo::Var * sumVar = getOutputStreamVar("sum");
    if (a.size() > mBixBits) a = bnc.Truncate(a, mBixBits);
    if (b.size() > mBixBits) b = bnc.Truncate(b, mBixBits);
    while(a.size() < mBixBits) {
        a.push_back(pb.createZeroes());
    }
    pablo::BixNum sum = bnc.AddModular(a, b);
    for (unsigned i = 0; i < mBixBits; i++) {
        pb.createAssign(pb.createExtract(sumVar, i), sum[i]);
    }
}

void Add_immediate::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::BixNum a = getInputStreamSet("a");
    pablo::Var * sumVar = getOutputStreamVar("sum");
    if (a.size() > mBixBits) a = bnc.Truncate(a, mBixBits);
    unsigned mask = (1u << mBixBits) - 1;
    while(a.size() < mBixBits) {
        a.push_back(pb.createZeroes());
    }
    pablo::BixNum sum = bnc.SubModular(a, mImmediate & mask);
    for (unsigned i = 0; i < mBixBits; i++) {
        pb.createAssign(pb.createExtract(sumVar, i), sum[i]);
    }
}

void Sub::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::BixNum a = getInputStreamSet("a");
    pablo::BixNum b = getInputStreamSet("b");
    pablo::Var * diffVar = getOutputStreamVar("diff");
    if (a.size() > mBixBits) a = bnc.Truncate(a, mBixBits);
    if (b.size() > mBixBits) b = bnc.Truncate(b, mBixBits);
    while(a.size() < mBixBits) {
        a.push_back(pb.createZeroes());
    }
    pablo::BixNum diff = bnc.SubModular(a, b);
    for (unsigned i = 0; i < mBixBits; i++) {
        pb.createAssign(pb.createExtract(diffVar, i), diff[i]);
    }
}

void Sub_immediate::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::BixNum a = getInputStreamSet("a");
    pablo::Var * diffVar = getOutputStreamVar("diff");
    if (a.size() > mBixBits) a = bnc.Truncate(a, mBixBits);
    unsigned mask = (1u << mBixBits) - 1;
    while(a.size() < mBixBits) {
        a.push_back(pb.createZeroes());
    }
    pablo::BixNum diff = bnc.SubModular(a, mImmediate & mask);
    for (unsigned i = 0; i < mBixBits; i++) {
        pb.createAssign(pb.createExtract(diffVar, i), diff[i]);
    }
}

void Mul_immediate::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::BixNum a = getInputStreamSet("a");
    pablo::Var * productVar = getOutputStreamVar("product");
    if (a.size() > mBixBits) a = bnc.Truncate(a, mBixBits);
    unsigned mask = (1u << mBixBits) - 1;
    while(a.size() < mBixBits) {
        a.push_back(pb.createZeroes());
    }
    pablo::BixNum product = bnc.MulModular(a, mImmediate & mask);
    for (unsigned i = 0; i < mBixBits; i++) {
        pb.createAssign(pb.createExtract(productVar, i), product[i]);
    }
}

#define compare_immediate(OP) \
void OP##_immediate::generatePabloMethod() { \
    pablo::PabloBuilder pb(getEntryScope()); \
    pablo::BixNumCompiler bnc(pb);\
    pablo::BixNum a = getInputStreamSet("a");\
    pablo::PabloAST * comparison = bnc.OP(a, mImmediate);\
    pablo::Var * rsltVar = getOutputStreamVar("rslt");\
    pb.createAssign(pb.createExtract(rsltVar, pb.getInteger(0)), comparison);\
}

compare_immediate(EQ)
compare_immediate(NEQ)
compare_immediate(UGT)
compare_immediate(UGE)
compare_immediate(ULT)
compare_immediate(ULE)

}
