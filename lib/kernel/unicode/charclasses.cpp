/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/unicode/charclasses.h>

#include <re/toolchain/toolchain.h>
#include <kernel/core/kernel_builder.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/adt/re_name.h>
#include <unicode/utf/utf_compiler.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <map>

using namespace cc;
using namespace kernel;
using namespace pablo;
using namespace re;
using namespace llvm;
using namespace UTF;

std::string makeSignature(const StreamSet * const basis, std::vector<re::CC *> & ccs) {
    std::string tmp;
    raw_string_ostream out(tmp);
    out << basis->getNumElements() << 'x' << basis->getFieldWidth();
    if (LLVM_LIKELY(!ccs.empty())) {
        char joiner = '[';
        for (const auto & set : ccs) {
            out << joiner;
            set->print(out);
            joiner = ',';
        }
        out << ']';
    }
    out << UTF::kernelAnnotation();
    out.flush();
    return tmp;
}

CharClassesKernel::CharClassesKernel(LLVMTypeSystemInterface & ts, 
                                     std::vector<re::CC *> ccs,
                                     StreamSet * BasisBits,
                                     StreamSet * CharClasses,
                                     BitMovementMode mode)
: CharClassesKernel(ts, makeSignature(BasisBits, ccs), std::move(ccs), BasisBits, CharClasses, mode) {

}

CharClassesKernel::CharClassesKernel(LLVMTypeSystemInterface & ts, std::string signature, std::vector<re::CC *> && ccs, StreamSet * BasisBits, StreamSet * CharClasses, BitMovementMode mode)
: PabloKernel(ts, "cc_" + getStringHash(signature) + UTF::kernelAnnotation() +
              pablo::BitMovementMode_string(mode)
, {Binding{"basis", BasisBits}}, {Binding{"charclasses", CharClasses}})
, mCCs(ccs)
, mSignature(signature)
, mBitMovement(mode) {

}

llvm::StringRef CharClassesKernel::getSignature() const {
    return mSignature;
}

void CharClassesKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    unsigned n = mCCs.size();

    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb, mBitMovement);
    std::vector<Var *> mpx;
    for (unsigned i = 0; i < n; i++) {
        mpx.push_back(pb.createVar("mpx_basis" + std::to_string(i), pb.createZeroes()));
    }
    unicodeCompiler.compile(mpx, mCCs);
    for (unsigned i = 0; i < mpx.size(); i++) {
        Extract * const r = pb.createExtract(getOutput(0), pb.getInteger(i));
        pb.createAssign(r, pb.createInFile(mpx[i]));
    }
}

ByteClassesKernel::ByteClassesKernel(LLVMTypeSystemInterface & ts,
                                     std::vector<re::CC *>  ccs,
                                     StreamSet * inputStream,
                                     StreamSet * CharClasses)
: ByteClassesKernel(ts, makeSignature(inputStream, ccs), std::move(ccs), inputStream, CharClasses) {

}

ByteClassesKernel::ByteClassesKernel(LLVMTypeSystemInterface & ts,
                                     std::string signature,
                                     std::vector<re::CC *> && ccs,
                                     StreamSet * inputStream,
                                     StreamSet * CharClasses)
: PabloKernel(ts, "bcc_" + getStringHash(signature)
, {Binding{"basis", inputStream}}, {Binding{"charclasses", CharClasses}})
, mCCs(ccs)
, mSignature(signature) {

}

StringRef ByteClassesKernel::getSignature() const {
    return mSignature;
}

void ByteClassesKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::unique_ptr<cc::CC_Compiler> ccc;

    auto basisBits = getInputStreamSet("basis");
    if (basisBits.size() == 1) {
        ccc = std::make_unique<cc::Direct_CC_Compiler>(basisBits[0]);
    } else {
        ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(basisBits);
    }
    unsigned n = mCCs.size();

    std::vector<PabloAST *> byte_class;
    for (unsigned i = 0; i < n; i++) {
        byte_class.push_back(ccc->compileCC(mCCs[i], pb));
    }
    for (unsigned i = 0; i < byte_class.size(); i++) {
        Extract * const r = pb.createExtract(getOutput(0), pb.getInteger(i));
        pb.createAssign(r, pb.createInFile(byte_class[i]));
    }
}
