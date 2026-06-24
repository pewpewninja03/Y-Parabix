/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/streamutils/string_insert.h>
#include <kernel/core/kernel_builder.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_ones.h>
#include <pablo/pe_var.h>
#include <pablo/pe_infile.h>
#include <pablo/pe_lookahead.h>
#include <pablo/bixnum/bixnum.h>
#include <re/cc/cc_compiler_target.h>
#include <re/cc/cc_compiler.h>
#include <re/alphabet/alphabet.h>
#include <llvm/Support/Compiler.h>

using namespace pablo;
using namespace llvm;

namespace kernel {

LLVM_READONLY std::string ZeroInsertName(const std::vector<unsigned> & insertAmts, const StreamSet * insertMarks) {
    std::string name = "ZeroInsertBixNum";
    for (auto a : insertAmts) {
        name += "_" + std::to_string(a);
    }
    if (insertMarks->getNumElements() < insertAmts.size()) {
        name += "_multiplexed";
    }
    return name;
}
    
ZeroInsertBixNum::ZeroInsertBixNum(LLVMTypeSystemInterface & ts, const std::vector<unsigned> & insertAmts,
                                       StreamSet * insertMarks, StreamSet * insertBixNum)
: PabloKernel(ts, "StringInsertBixNum" + Kernel::getStringHash(ZeroInsertName(insertAmts, insertMarks)),
              {Binding{"insertMarks", insertMarks}},
              {Binding{"insertBixNum", insertBixNum}})
, mInsertAmounts(insertAmts)
, mMultiplexing(insertMarks->getNumElements() < insertAmts.size())
, mBixNumBits(insertBixNum->getNumElements())
, mSignature(ZeroInsertName(insertAmts, insertMarks)) {

}

void ZeroInsertBixNum::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    std::vector<PabloAST *> insertMarks = getInputStreamSet("insertMarks");
    std::vector<PabloAST *> bixNum(mBixNumBits, pb.createZeroes());
    Var * insertVar = getOutputStreamVar("insertBixNum");
    for (unsigned i = 0; i < mInsertAmounts.size(); i++) {
        PabloAST * stringMarks = mMultiplexing ? bnc.EQ(insertMarks, i + 1) : insertMarks[i];
        for (unsigned j = 0; j < mBixNumBits; j++) {
            if ((mInsertAmounts[i] >> j) & 1) {
                bixNum[j] = pb.createOr(bixNum[j], stringMarks);
            }
        }
    }
    for (unsigned j = 0; j < mBixNumBits; j++) {
        pb.createAssign(pb.createExtract(insertVar, pb.getInteger(j)), bixNum[j]);
    }
}

static std::string StringReplaceName(const std::vector<std::string> & insertStrs, const StreamSet * insertMarks, int markOffset) {
    std::string name;
    raw_string_ostream nm(name);
    nm << "StringReplaceBixNum";
    if (insertMarks->getNumElements() < insertStrs.size()) {
        nm << insertMarks->getNumElements() << ':';
    }
    nm << markOffset;
    for (const auto & s : insertStrs) {
        name += "_" + s;
    }
    nm.flush();
    return name;
}

StringReplaceKernel::StringReplaceKernel(LLVMTypeSystemInterface & ts, const std::vector<std::string> & insertStrs,
                                         StreamSet * basis, StreamSet * spreadMask,
                                         StreamSet * insertMarks, StreamSet * runIndex,
                                         StreamSet * output, int markOffset)
: StringReplaceKernel(ts, StringReplaceName(insertStrs, insertMarks, markOffset),
                      insertStrs, basis, spreadMask, insertMarks, runIndex, output, markOffset) {

}

StringReplaceKernel::StringReplaceKernel(LLVMTypeSystemInterface & ts, std::string && signature,
                                         const std::vector<std::string> & insertStrs,
                                         StreamSet * basis, StreamSet * spreadMask,
                                         StreamSet * insertMarks, StreamSet * runIndex,
                                         StreamSet * output, int markOffset)
: PabloKernel(ts, "StringReplaceBixNum" + Kernel::getStringHash(signature),
             {Binding{"basis", basis}, Binding{"spreadMask", spreadMask},
              Binding{"insertMarks", insertMarks, FixedRate(1), LookAhead(1U << (runIndex->getNumElements()))},
              Binding{"runIndex", runIndex}},
             {Binding{"output", output}})
, mInsertStrings(insertStrs)
, mMultiplexing(insertMarks->getNumElements() < insertStrs.size())
, mMarkOffset(markOffset)
, mSignature(std::move(signature)) {

}



void StringReplaceKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    PabloAST * spreadMask = getInputStreamSet("spreadMask")[0];
    std::vector<PabloAST *> insertMarks = getInputStreamSet("insertMarks");
    std::vector<PabloAST *> runIndex = getInputStreamSet("runIndex");
    Var * output = getOutputStreamVar("output");
    std::unique_ptr<cc::CC_Compiler> ccc;
    ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(runIndex);
    PabloAST * runMask = pb.createInFile(pb.createNot(spreadMask));
    std::vector<PabloAST *> insertSpans(mInsertStrings.size());
    for (unsigned i = 0; i < mInsertStrings.size(); i++) {
        unsigned lookahead = 0;
        if (mMarkOffset > 0) {
            lookahead = mMarkOffset;
        } else if (mMarkOffset < 0) {
            lookahead = mInsertStrings[i].size() + 1 + mMarkOffset;
        }
        PabloAST * stringStart;
        if (lookahead == 0) {
            stringStart = mMultiplexing ? bnc.EQ(insertMarks, i + 1) : insertMarks[i];
        } else if (mMultiplexing) {
            BixNum ahead(insertMarks.size());
            for (unsigned j = 0; j < insertMarks.size(); j++) {
                ahead[j] = pb.createLookahead(insertMarks[j], lookahead);
            }
            stringStart = bnc.EQ(ahead, i + 1);
        } else {
            stringStart = pb.createLookahead(insertMarks[i], lookahead);
        }
        PabloAST * span = pb.createMatchStar(stringStart, runMask);
        insertSpans[i] = pb.createAnd(span, runMask);
    }
    for (unsigned bit = 0; bit < 8; bit++) {
        PabloAST * updated = basis[bit];
        for (unsigned i = 0; i < mInsertStrings.size(); i++) {
            re::CC * bitCC = re::makeCC(&cc::Byte);
            for (unsigned j = 0; j < mInsertStrings[i].size(); j++) {
                if ((mInsertStrings[i][j] >> bit) & 1) {
                    bitCC->insert(j);
                }
            }
            PabloAST * ccStrm = ccc->compileCC(bitCC, pb);
            updated = pb.createSel(insertSpans[i], ccStrm, updated);
        }
        pb.createAssign(pb.createExtract(output, pb.getInteger(bit)), updated);
    }
}
}
