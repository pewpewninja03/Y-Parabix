#include <kernel/re/regexp_kernel.h>
#include <kernel/core/kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <pablo/builder.hpp>
#include <pablo/pe_var.h>
#include <pablo/pe_zeroes.h>
#include <re/adt/adt.h>
#include <re/analysis/re_analysis.h>
#include <re/printer/re_printer.h>
#include <re/toolchain/toolchain.h>
#include <re/cc/cc_compiler.h>         // for CC_Compiler
#include <re/cc/cc_compiler_target.h>
#include <re/compile/re_compiler.h>
#include <re/transforms/name_intro.h>

using namespace re;
using namespace pablo;
using namespace kernel;

RE_CompilerContext::RE_CompilerContext() : mCodeUnitAlphabet(nullptr), mCodeUnitStream(nullptr),
    mBarrierStream(nullptr), mIndexingAlphabet(nullptr), mIndexStream(nullptr),
    mCombiningType(RE_CombiningType::None), mCombiningStream(nullptr) {}

void RE_CompilerContext::setCodeUnitContext(const cc::Alphabet * a, StreamSet * basis) {
    mCodeUnitAlphabet = a;
    mCodeUnitStream = basis;
    addAlphabet(a, basis);
}

void RE_CompilerContext::setBarrier(kernel::StreamSet * s) {
    mBarrierStream = s;
}

void RE_CompilerContext::setIndexingContext(const cc::Alphabet * a, StreamSet * s) {
    mIndexingAlphabet = a;
    mIndexStream = s;
}

void RE_CompilerContext::addAlphabet(const cc::Alphabet * a, StreamSet * basis) {
    mAlphabets.emplace_back(a, basis);
}

void RE_CompilerContext::addExternal(std::string extName, ExternalStream s) {
    mExternals.emplace(extName, s);
}

RE_Kernel::RE_Kernel(LLVMTypeSystemInterface & ts, RE_CompilerContext & ctxt, RE * re, StreamSet * results)
: PabloKernel(ts, makeSignature(ctxt, re), makeInputBindings(ctxt, re), makeOutputBindings(re, results), {}, {}),
mContext(ctxt), mRE(re) {
    addAttribute(InfrequentlyUsed());
}

std::string RE_Kernel::makeSignature(RE_CompilerContext & ctxt, RE * re) {
    std::string signature;
    llvm::raw_string_ostream sigstrm(signature);
    sigstrm << AnnotateWithREflags("RE");
    if (ctxt.mBarrierStream) {
        sigstrm << "-B";
    }    
    if (ctxt.mIndexStream) {
        sigstrm << "+X";
    }
    for (auto & a : ctxt.mAlphabets) {
        sigstrm << '!' << a.second->getNumElements() << 'x' << a.second->getFieldWidth();
    }
    std::vector<std::string> externalList;
    for (const auto & e : ctxt.mExternals) {
        auto name = e.first;
        auto ext = e.second;
        if (ext.kind == ExternalStreamKind::ZeroWidth) {
            sigstrm << "+Z"; 
        } else if (ext.kind == ExternalStreamKind::FixedLength) {
            sigstrm << "+F"; 
        } else if (ext.kind == ExternalStreamKind::StartIndexed) {
            sigstrm << "+S"; 
        } else {
            sigstrm << "+E";
        }
        externalList.push_back(name);
        sigstrm << name << ext.offset;
    }
    if (ctxt.mCombiningType == RE_CombiningType::Exclude) {
        sigstrm << "&~";
    } else if (ctxt.mCombiningType == RE_CombiningType::Include) {
        sigstrm << "|=";
    }
    std::string canon_string = Printer_RE::PrintRE(canonicalizeExternals(re, externalList));
    sigstrm << ":" << Kernel::getStringHash(canon_string);
    sigstrm.flush();
    return signature;
}

unsigned round_up_to_blocksize(int lgth) {
    unsigned lookahead_blocks = (codegen::BlockSize - 1 + lgth)/codegen::BlockSize;
    return lookahead_blocks * codegen::BlockSize;
}

// TODO:  limit the alphabets and externals to those in the RE top-level
Bindings RE_Kernel::makeInputBindings(RE_CompilerContext & ctxt, RE * re) {
    Bindings externalBindings;
    if (ctxt.mBarrierStream) {
        externalBindings.emplace_back("mBarrier", ctxt.mBarrierStream);
    }
    if (ctxt.mIndexStream) {
        externalBindings.emplace_back("mIndexing", ctxt.mIndexStream);
    }
    for (auto & a : ctxt.mAlphabets) {
        externalBindings.emplace_back(a.first->getName() + "_basis", a.second);
    }
    std::vector<std::string> externalList;
    for (const auto & e : ctxt.mExternals) {
        auto name = e.first;
        auto ext = e.second;
        if (ext.kind == StartIndexed) {
            unsigned blk_ahead = round_up_to_blocksize(ext.offset);
            externalBindings.emplace_back(name, ext.extStream, FixedRate(), LookAhead(blk_ahead));
        } else {
            externalBindings.emplace_back(name, ext.extStream);
        }
    }
    if (ctxt.mCombiningType != RE_CombiningType::None) {
        externalBindings.emplace_back("toCombine", ctxt.mCombiningStream, FixedRate(), Add1());
    }
    return externalBindings;
}

Bindings RE_Kernel::makeOutputBindings(RE * re, StreamSet * results) {
    if (grepOffset(re) == 0) {
        return {Binding{"matches", results}};
    }
    return {Binding{"matches", results, FixedRate(), Add1()}};    
}

void RE_Kernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * barrier = nullptr;
    if (mContext.mBarrierStream) {
        barrier = pb.createExtract(getInputStreamVar("mBarrier"), pb.getInteger(0));
    }
    RE_Compiler re_compiler(getEntryScope(), barrier, mContext.mCodeUnitAlphabet);
    for (auto & a : mContext.mAlphabets) {
        auto basis = getInputStreamSet(a.first->getName() + "_basis");
        re_compiler.addAlphabet(a.first, basis);
    }
    if (mContext.mIndexStream) {
        PabloAST * idxStrm = pb.createExtract(getInputStreamVar("mIndexing"), pb.getInteger(0));
        re_compiler.setIndexing(&cc::Unicode, idxStrm);
    }
    for (auto & e : mContext.mExternals) {
        auto name = e.first;
        auto ext = e.second;
        PabloAST * extStrm = pb.createExtract(getInputStreamVar(name), pb.getInteger(0));
        unsigned offset = ext.offset;
        std::pair<int, int> lgthRange;
        bool fromFirst = false;
        if (ext.kind == ExternalStreamKind::ZeroWidth) {
            offset = 1;
            lgthRange = {0, 0};
        } else if (ext.kind == ExternalStreamKind::FixedLength) {
            lgthRange = {0, 0};
        } else if (ext.kind == ExternalStreamKind::StartIndexed) {
            fromFirst = true;
            lgthRange = {1, 1};
        } else {
            lgthRange = {1, 1000};
        }
        re_compiler.addPrecompiled(name, RE_Compiler::ExternalStream(RE_Compiler::Marker(extStrm, offset), lgthRange, fromFirst));
    }
    Var * const final_matches = pb.createVar("final_matches", pb.createZeroes());
    RE_Compiler::Marker matches = re_compiler.compileRE(mRE);
    PabloAST * matchResult = matches.stream();
    //if (matches.offset() != mOffset) {
        //errs() << Printer_RE::PrintRE(mContext.mRE) <<"\n mOffset = " << mOffset << "\n";
        //report_fatal_error("matches.offset() != mOffset");
    //}
    pb.createAssign(final_matches, matchResult);
    Var * const output = pb.createExtract(getOutputStreamVar("matches"), pb.getInteger(0));
    pb.createAssign(output, final_matches);
}

