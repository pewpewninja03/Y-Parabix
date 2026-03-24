#include <kernel/re/regexp_kernel.h>
#include <kernel/core/kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/bitwise/bixlogic.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/streamutils/streams_merge.h>
#include <kernel/unicode/boundary_kernels.h>
#include <kernel/unicode/charclasses.h>
#include <kernel/unicode/UCD_property_kernel.h>
#include <pablo/builder.hpp>
#include <pablo/pe_var.h>
#include <pablo/pe_zeroes.h>
#include <re/adt/adt.h>
#include <re/alphabet/alphabet.h>
#include <re/alphabet/multiplex_CCs.h>
#include <re/analysis/re_analysis.h>
#include <re/analysis/collect_ccs.h>
#include <re/analysis/re_name_gather.h>
#include <re/analysis/capture-ref.h>
#include <re/printer/re_printer.h>
#include <re/toolchain/toolchain.h>
#include <re/unicode/regex_passes.h>
#include <re/transforms/to_utf8.h>
#include <re/transforms/re_multiplex.h>
#include <re/transforms/expand_permutes.h>
#include <re/transforms/name_lookaheads.h>
#include <re/transforms/reference_transform.h>
#include <re/transforms/remove_nullable.h>
#include <re/transforms/variable_alt_promotion.h>
#include <re/unicode/boundaries.h>
#include <re/unicode/resolve_properties.h>
#include <re/cc/cc_compiler.h>         // for CC_Compiler
#include <re/cc/cc_compiler_target.h>
#include <re/compile/re_compiler.h>
#include <re/transforms/name_intro.h>
#include <unicode/data/PropertyObjects.h>
#include <unicode/data/PropertyObjectTable.h>
#include <unicode/utf/utf_compiler.h>

using namespace re;
using namespace pablo;
using namespace kernel;
using namespace llvm;

RE_CompilerContext::RE_CompilerContext() : mCodeUnitAlphabet(nullptr), mCodeUnitStream(nullptr),
    mBarrierStream(nullptr), mLengthAlphabet(nullptr), mIndexStream(nullptr),
    mCombiningType(RE_CombiningType::None), mCombiningStream(nullptr) {}

void RE_CompilerContext::setCodeUnitContext(const cc::Alphabet * a, StreamSet * basis) {
    mCodeUnitAlphabet = a;
    mLengthAlphabet = a; // default
    mCodeUnitStream = basis;
    addAlphabet(a, basis);
}

void RE_CompilerContext::setBarrier(kernel::StreamSet * s) {
    mBarrierStream = s;
}

void RE_CompilerContext::setIndexingContext(const cc::Alphabet * a, StreamSet * s) {
    mLengthAlphabet = a;
    mIndexStream = s;
}

void RE_CompilerContext::addAlphabet(const cc::Alphabet * a, StreamSet * basis) {
    mAlphabets.emplace_back(a, basis);
}

void RE_CompilerContext::addExternal(std::string extName, ExternalStream s) {
    mExternals.emplace(extName, s);
}

void RE_CompilerContext::setCombiningStream(kernel::StreamSet * s, RE_CombiningType k) {
    mCombiningStream = s;
    mCombiningType = k;
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
    std::set<std::string> localExternals;
    std::set<std::string> localAlphabets;
    gatherExternals(re, localExternals, localAlphabets);
    for (auto & a : ctxt.mAlphabets) {
        auto alphaName = a.first->getName();
        if (localAlphabets.count(alphaName) == 1) {
            sigstrm << '!' << a.second->getNumElements() << 'x' << a.second->getFieldWidth();
        }
    }
    std::vector<std::string> externalList;
    for (const auto & e : ctxt.mExternals) {
        auto name = e.first;
        if (localExternals.count(name) == 1) {
            auto ext = e.second;
            if (ext.kind == ExternalStreamKind::ZeroWidth) {
                sigstrm << "+Z";
            } else if (ext.kind == ExternalStreamKind::FixedLength) {
                sigstrm << "+F";
                sigstrm << ext.lgthRange.second;
                if (ext.offset == 1) sigstrm << "o";
            } else if (ext.kind == ExternalStreamKind::StartIndexed) {
                sigstrm << "+S";
                sigstrm << ext.offset;
            } else {
                sigstrm << "+E";
                sigstrm << ext.lgthRange.first << "-" << ext.lgthRange.second;
                if (ext.offset == 1) sigstrm << "o";
            }
            externalList.push_back(name);
        }
    }
    if (ctxt.mCombiningType == RE_CombiningType::Exclude) {
        sigstrm << "&~";
    } else if (ctxt.mCombiningType == RE_CombiningType::Include) {
        sigstrm << "|=";
    }
    std::string canon_string = Printer_RE::PrintRE(canonicalizeExternals(re, externalList));
    //llvm::errs() << "RE_Kernel: " << canon_string << "\n";
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
    std::set<std::string> localExternals;
    std::set<std::string> localAlphabets;
    gatherExternals(re, localExternals, localAlphabets);
    Bindings externalBindings;
    if (ctxt.mBarrierStream) {
        externalBindings.emplace_back("mBarrier", ctxt.mBarrierStream);
    }
    if (ctxt.mIndexStream) {
        externalBindings.emplace_back("mIndexing", ctxt.mIndexStream);
    }
    for (auto & a : ctxt.mAlphabets) {
        auto alphaName = a.first->getName();
        //llvm::errs() << "alphaName: " << alphaName << " count = " << localAlphabets.count(alphaName) << "\n";
        if (localAlphabets.count(alphaName) == 1) {
            externalBindings.emplace_back(alphaName + "_basis", a.second);
        }
    }
    std::vector<std::string> externalList;
    for (const auto & e : ctxt.mExternals) {
        auto name = e.first;
        if (localExternals.count(name) == 1) {
            //llvm::errs() << "RE_Kernel - adding external: " << name << "\n";
            auto ext = e.second;
            if (ext.kind == StartIndexed) {
                unsigned blk_ahead = round_up_to_blocksize(ext.offset);
                externalBindings.emplace_back(name, ext.extStream, FixedRate(), LookAhead(blk_ahead));
            } else {
                externalBindings.emplace_back(name, ext.extStream);
            }
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
    std::set<std::string> localExternals;
    std::set<std::string> localAlphabets;
    gatherExternals(mRE, localExternals, localAlphabets);
    PabloAST * barrier = nullptr;
    if (mContext.mBarrierStream) {
        barrier = pb.createExtract(getInputStreamVar("mBarrier"), pb.getInteger(0));
    }
    RE_Compiler re_compiler(getEntryScope(), barrier, mContext.mCodeUnitAlphabet);
    for (auto & a : mContext.mAlphabets) {
        auto alphaName = a.first->getName();
        if (localAlphabets.count(alphaName) == 1) {
            auto basis = getInputStreamSet(alphaName + "_basis");
            re_compiler.addAlphabet(a.first, basis);
        }
    }
    if (mContext.mIndexStream) {
        PabloAST * idxStrm = pb.createExtract(getInputStreamVar("mIndexing"), pb.getInteger(0));
        re_compiler.setIndexing(&cc::Unicode, idxStrm);
    }
    for (auto & e : mContext.mExternals) {
        auto name = e.first;
        if (localExternals.count(name) == 1) {
            auto ext = e.second;
            PabloAST * extStrm = pb.createExtract(getInputStreamVar(name), pb.getInteger(0));
            unsigned offset = ext.offset;
            bool fromFirst = false;
            if (ext.kind == ExternalStreamKind::StartIndexed) {
                fromFirst = true;
            }
            re_compiler.addPrecompiled(name, RE_Compiler::ExternalStream(extStrm, offset, ext.lgthRange, fromFirst));
        }
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

    PabloAST * value = nullptr;
    if (mContext.mCombiningType == RE_CombiningType::None) {
        value = final_matches;
    } else {
        PabloAST * toCombine = pb.createExtract(getInputStreamVar("toCombine"), pb.getInteger(0));
        if (mContext.mCombiningType == RE_CombiningType::Exclude) {
            value = pb.createAnd(toCombine, pb.createNot(final_matches), "toCombine");
        } else {
            value = pb.createOr(toCombine, final_matches, "toCombine");
        }
    }
    pb.createAssign(output, value);
}

PabloAST * matchDistanceCheck(PabloBuilder & b, unsigned distance, std::vector<PabloAST *> basis1, std::vector<PabloAST *> basis2) {
    PabloAST * differ = b.createZeroes();
    for (unsigned i = 0; i < basis1.size(); i++) {
        PabloAST * advanced = b.createAdvance(basis1[i], distance);
        differ = b.createOr(differ, b.createXor(advanced, basis2[i]));
    }
    return differ;
}

void FixedDistanceMatchesKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    auto basis = getInputStreamSet("Basis");
    Var * mismatch = pb.createVar("mismatch", pb.createZeroes());
    if (mHasCheckStream) {
        auto ToCheck = getInputStreamSet("ToCheck")[0];
        auto it = pb.createScope();
        pb.createIf(ToCheck, it);
        PabloAST * differ = matchDistanceCheck(it, mMatchDistance, basis, basis);
        it.createAssign(mismatch, it.createAnd(ToCheck, differ));
    } else {
        pb.createAssign(mismatch, matchDistanceCheck(pb, mMatchDistance, basis, basis));
    }
    Var * const MatchVar = getOutputStreamVar("Matches");
    pb.createAssign(pb.createExtract(MatchVar, pb.getInteger(0)), pb.createNot(mismatch, "matches"));
}

FixedDistanceMatchesKernel::FixedDistanceMatchesKernel (LLVMTypeSystemInterface & ts, unsigned distance, StreamSet * Basis, StreamSet * Matches, StreamSet * ToCheck)
: PabloKernel(ts, "Distance_" + std::to_string(distance) + "_Matches_" + std::to_string(Basis->getNumElements()) + "x1" + (ToCheck == nullptr ? "" : "_withCheck"),
// inputs
{Binding{"Basis", Basis}},
// output
{Binding{"Matches", Matches}}), mMatchDistance(distance), mHasCheckStream(ToCheck != nullptr) {
    if (mHasCheckStream) {
        mInputStreamSets.push_back({"ToCheck", ToCheck});
    }
}

void CodePointMatchKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UCD::PropertyObject * propObj = UCD::getPropertyObject(mProperty);
    if (UCD::CodePointPropertyObject * p = dyn_cast<UCD::CodePointPropertyObject>(propObj)) {
        const UCD::UnicodeSet & nullSet = p->GetNullSet();
        std::vector<UCD::UnicodeSet> & xfrm_ccs = p->GetBitTransformSets();
        UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
        std::vector<Var *> xfrm_vars;
        for (unsigned i = 0; i < xfrm_ccs.size(); i++) {
            xfrm_vars.push_back(pb.createVar("xfrm_cc_" + std::to_string(i), pb.createZeroes()));
        }
        Var * nullVar = nullptr;
        if (!nullSet.empty()) {
            xfrm_ccs.push_back(nullSet);
            nullVar = pb.createVar("null_set", pb.createZeroes());
            xfrm_vars.push_back(nullVar);
        }
        unicodeCompiler.compile(xfrm_vars, xfrm_ccs);
        std::vector<PabloAST *> basis = getInputStreamSet("Basis");
        std::vector<PabloAST *> transformed(basis.size());
        for (unsigned i = 0; i < basis.size(); i++) {
            if (i < xfrm_vars.size()) {
                transformed[i] = pb.createXor(xfrm_vars[i], basis[i]);
            } else {
                transformed[i] = basis[i];
            }
        }
        PabloAST * mismatch;
        bool involution = ((mProperty == UCD::bpb) || (mProperty == UCD::bmg));
        if (involution) {
            mismatch = matchDistanceCheck(pb, mMatchDistance, transformed, basis);
        } else {
            mismatch = matchDistanceCheck(pb, mMatchDistance, transformed, transformed);
        }
        if (!nullSet.empty()) {
            mismatch = pb.createOr(mismatch, nullVar);
        }
        PabloAST * matches = pb.createInFile(pb.createNot(mismatch));
        Var * const MatchVar = getOutputStreamVar("Matches");
        pb.createAssign(pb.createExtract(MatchVar, pb.getInteger(0)), matches);
    } else {
        llvm::report_fatal_error("Expecting codepoint property");
    }
}

CodePointMatchKernel::CodePointMatchKernel (LLVMTypeSystemInterface & ts, UCD::property_t prop, unsigned distance, StreamSet * Basis, StreamSet * Matches)
: PabloKernel(ts, getPropertyEnumName(prop) + "_dist_" + std::to_string(distance) + "_Matches_" + std::to_string(Basis->getNumElements()) + "x1" + UTF::kernelAnnotation(),
// inputs
{Binding{"Basis", Basis}},
// output
{Binding{"Matches", Matches}}),
    mMatchDistance(distance),
    mProperty(prop) {
}

FixedMatchSpansKernel::FixedMatchSpansKernel(LLVMTypeSystemInterface & ts, unsigned length, unsigned offset, StreamSet * MatchMarks, StreamSet * MatchSpans)
: PabloKernel(ts, "FixedMatchSpansKernel" + std::to_string(MatchMarks->getNumElements()) + "x1_by" + std::to_string(length) + '@' + std::to_string(offset),
{Binding{"MatchMarks", MatchMarks, FixedRate(1), LookAhead(round_up_to_blocksize(length))}}, {Binding{"MatchSpans", MatchSpans}}),
mMatchLength(length), mOffset(offset) {
}

void FixedMatchSpansKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * marks = pb.createExtract(getInputStreamVar("MatchMarks"), pb.getInteger(0));
    Var * matchSpansVar = getOutputStreamVar("MatchSpans");
    // starts of all the matches
    PabloAST * starts = pb.createLookahead(marks, mMatchLength + mOffset - 1);
    // now find all consecutive positions within mMatchLength of any start.
    unsigned consecutiveCount = 1;
    PabloAST * consecutive = starts;
    for (unsigned i = 1; i <= mMatchLength/2; i *= 2) {
        consecutiveCount += i;
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, i),
                                  "consecutive" + std::to_string(consecutiveCount));
    }
    if (consecutiveCount < mMatchLength) {
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, mMatchLength - consecutiveCount),
                                  "consecutive" + std::to_string(mMatchLength));
    }
    pb.createAssign(pb.createExtract(matchSpansVar, 0), consecutive);
}

void LongestSpan::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * pfxStrm = getInputStreamSet("pfxStrm")[0];
    PabloAST * endBack = getInputStreamSet("endBack")[0];
    PabloAST * matchEnd = getInputStreamSet("matchEnd")[0];
    PabloAST * pfxStart = pb.createAnd(pb.createLookahead(pfxStrm, mPfxOffset), pb.createLookahead(endBack, mPfxOffset));
    PabloAST * longestEnd = pb.createAnd(matchEnd, pb.createNot(endBack));
    if (mEndOffset != 0) {
        longestEnd = pb.createAdvance(longestEnd, 1);
    }
    PabloAST * spans = pb.createIntrinsicCall(pablo::Intrinsic::SpanUpTo, {pfxStart, longestEnd});
    writeOutputStreamSet("spans", std::vector<PabloAST*>{spans});
}

LongestSpan::LongestSpan (LLVMTypeSystemInterface & ts, unsigned pfxOffset, unsigned endOffset, 
    StreamSet * pfxStrm, StreamSet * endBack, StreamSet * matchEnd, StreamSet * spans)
: PabloKernel(ts, "LongestSpan_" + std::to_string(pfxOffset) + ":" + std::to_string(endOffset),
// inputs
{Binding{"pfxStrm", pfxStrm, FixedRate(1), LookAhead(pfxOffset)},
 Binding{"endBack", endBack, FixedRate(1), LookAhead(pfxOffset)},
 Binding{"matchEnd", matchEnd}},
// output
{Binding{"spans", spans}}), mPfxOffset(pfxOffset),  mEndOffset(endOffset) {
    
}


void RE_PipelineBuilder::addExternal(std::string extName, ExternalStream s) {
    //llvm::errs() << "addExternal(" << extName << ", ";
    //if (s.kind == ExternalStreamKind::StartIndexed) llvm::errs() << "StartIndexed";
    //if (s.kind == ExternalStreamKind::ZeroWidth) llvm::errs() << "ZeroWidth";
    //if (s.kind == ExternalStreamKind::FixedLength) llvm::errs() << "FixedLength";
    //if (s.kind == ExternalStreamKind::EndIndexed) llvm::errs() << "EndIndexed";
    //llvm::errs() << ", (" << s.lgthRange.first << ", " << s.lgthRange.second << "), " << s.offset << ")\n";
    mCtxt.mExternals.emplace(extName, s);
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        mPB.captureBitstream(extName, s.extStream);
    }
}

void RE_PipelineBuilder::compileProperty(PropertyExpression * pe) {
    StreamSet * pStrm  = mPB.CreateStreamSet(1);
    UnicodePropertyLogic(mPB, pe, mCtxt.mCodeUnitStream, mCtxt.mIndexStream, pStrm);
    std::string propName = pe->getFullName();
    if (pe->getKind() == re::PropertyExpression::Kind::Codepoint) {
        addExternal(propName, ExternalStream{ExternalStreamKind::FixedLength, 0, {1, 1}, pStrm});
    } else { //PropertyExpression::Kind::Boundary
        UCD::property_t prop = static_cast<UCD::property_t>(pe->getPropertyCode());
        if (prop == UCD::g) propName = "\\b{g}";
        else if (prop == UCD::w) propName = "\\b{w}";
        addExternal(propName, ExternalStream{ExternalStreamKind::ZeroWidth, 1, {0, 0}, pStrm});
    }
}

void RE_PipelineBuilder::prepareExternals(RE * re) {
    std::set<re::Name *> externals;
    re::gatherNames(re, externals);
    for (const auto & e : externals) {
        compileExternal(e);
    }
}

void RE_PipelineBuilder::compileExternal(Name * n) {
    auto name = n->getFullName();
    auto f = mCtxt.mExternals.find(name);
    if (f != mCtxt.mExternals.end()) {
        // already compiled.
        return;
    }
    RE * defn = n->getDefinition();
    if (defn == nullptr) {
        llvm::report_fatal_error("RE_PipelineBuilder: Name is undefined.");
    }
    if (PropertyExpression * pe = dyn_cast<PropertyExpression>(defn)) {
        compileProperty(pe);
        return;
    }
    //  Need to compile this defn.   Make sure all referenced externals
    //  are compiled first.
    prepareExternals(defn);
    //
    // The defining expression can now be compiled.  In most cases,
    // a single RE_Kernel can be used.   However, for lookahead
    // assertions, special treatment is required.
    unsigned amt = NamedLookAheadAmount(n, *mCtxt.mLengthAlphabet);
    unsigned offset = grepOffset(defn);
    if (amt == 0) {
        // Not a lookahead; compile using a single kernel call.
        StreamSet * extStrm = mPB.CreateStreamSet(1);
        mPB.CreateKernelFamilyCall<RE_Kernel>(mCtxt, defn, extStrm);
        auto r = getLengthRange(defn, mCtxt.mCodeUnitAlphabet);
        if (r.second == 0) {
            addExternal(name, ExternalStream{ExternalStreamKind::ZeroWidth, 1, r, extStrm});
        } else if (r.first == r.second) {
            addExternal(name, ExternalStream{ExternalStreamKind::FixedLength, offset, r, extStrm});
        } else {
            addExternal(name, ExternalStream{ExternalStreamKind::EndIndexed, offset, r, extStrm});
        }
    } else {
        Assertion * a = llvm::cast<Assertion>(defn);
        RE * asserted = a->getAsserted();
        StreamSet * assertedStrm = mPB.CreateStreamSet(1);
        mPB.CreateKernelFamilyCall<RE_Kernel>(mCtxt, asserted, assertedStrm);
        if (a->getSense() == Assertion::Sense::Negative) {
            StreamSet * negatedStrm = mPB.CreateStreamSet(1);
            Invert(mPB, assertedStrm, negatedStrm);
            assertedStrm = negatedStrm;
        }
        auto r = getLengthRange(asserted, mCtxt.mLengthAlphabet);
        if (r.first == r.second) {
            // Fixed length lookaheads can be stored directly.
            unsigned lgth = static_cast<unsigned>(r.second);
            addExternal(name, ExternalStream{ExternalStreamKind::StartIndexed, lgth, r, assertedStrm});
        } else {
            // Apply the logic of matching a lookahead with a unique prefix.  
            // Match positions for the lookahead (assertedStrm) are shifted back to the 
            // prior prefix position.   This is implemented by making a mask stream
            // consisting of full matches to the lookahead (assertedStrm) and matches
            // to the unique prefix.  An indexed shift back of asserted matches
            // moves the match to the position of its unique fixed length prefix.
            if (!isa<Seq>(asserted) || !isa<Name>(cast<Seq>(asserted)->front())) {
                llvm::report_fatal_error("Expecting named unique prefix lookahead");
            }
            Name * prefName = cast<Name>(cast<Seq>(asserted)->front());
            auto f = mCtxt.mExternals.find(prefName->getFullName());
            StreamSet * prefStrm = f->second.extStream;
            StreamSet * maskStrm = mPB.CreateStreamSet(1);
            OrCombine(mPB, prefStrm, assertedStrm, maskStrm);
            StreamSet * assertedBack = mPB.CreateStreamSet(1);
            mPB.CreateKernelCall<IndexedShiftBack>(maskStrm, assertedStrm, assertedBack);
            StreamSet * extStrm = mPB.CreateStreamSet(1);
            AndCombine(mPB, prefStrm, assertedBack, extStrm);
            addExternal(name, ExternalStream{ExternalStreamKind::StartIndexed, amt, r, extStrm});
        }
    }
}

void RE_PipelineBuilder::matchSearchPipeline(RE * re, StreamSet * results) {
    mRE = prepareRE(re);
    mRE = processReferences(mRE);
    prepareExternals(mRE);
    mPB.CreateKernelFamilyCall<RE_Kernel>(mCtxt, mRE, results);
}

void RE_PipelineBuilder::matchSpanPipeline(RE * re, StreamSet * matches, StreamSet * spans) {
    mRE = prepareRE(re);
    mRE = processReferences(mRE);
    mRE = spanFactoring(mRE);
    prepareExternals(mRE);
    mPB.CreateKernelFamilyCall<RE_Kernel>(mCtxt, mRE, matches);
    auto subsetRE = emptyMatchElimination(mRE);
    getSpan(subsetRE, spans);
}

void RE_PipelineBuilder::getSpan(RE * re, StreamSet * spans) {
    if (Alt * a = dyn_cast<Alt>(re)) {
        if (a->size() == 1) {
            getSpan(a->front(), spans);
        } else {
            std::vector<StreamSet *> allSpans;
            for (auto & e : *a) {
                StreamSet * span = mPB.CreateStreamSet(1);
                getSpan(e, span);
                allSpans.push_back(span);
            }
            mPB.CreateKernelCall<StreamsMerge>(allSpans, spans);
        }
    } else if (Name * n = dyn_cast<Name>(re)) {
        auto name = n->getFullName();
        //llvm::errs() << "getSpan finding external: " << name << "\n";
        auto f = mCtxt.mExternals.find(name);
        assert(f != mCtxt.mExternals.end());
        auto matchEnd = f->second.extStream;
        auto minlgth = f->second.lgthRange.first;
        auto endOffset = f->second.offset;
        auto f2 = mUPnamer.mNameMap.find(name);
        if (f2 != mUPnamer.mNameMap.end()) {
            auto namedRE = f2->second;
            re::Name * prefixName = cast<re::Name>(cast<re::Seq>(namedRE)->front());
            std::string prefixStr = prefixName->getFullName();
            auto fp = mCtxt.mExternals.find(prefixStr);
            assert(fp != mCtxt.mExternals.end());
            auto pfxStrm = fp->second.extStream;
            auto pfxLgth = fp->second.lgthRange.first;
            auto pfxOffset = fp->second.offset;
            StreamSet * maskStrm = mPB.CreateStreamSet(1);
            OrCombine(mPB, pfxStrm, matchEnd, maskStrm);
            StreamSet * endBack = mPB.CreateStreamSet(1);
            mPB.CreateKernelCall<IndexedShiftBack>(maskStrm, matchEnd, endBack);
            mPB.CreateKernelCall<LongestSpan>(pfxLgth + pfxOffset - 1, endOffset, pfxStrm, endBack, matchEnd, spans);
        } else {
            mPB.CreateKernelFamilyCall<FixedMatchSpansKernel>(minlgth, endOffset, matchEnd, spans);
        }
        if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
            auto spanName = name + "Span";
            mPB.captureBitstream(spanName, spans);
        }
    } else {
        StreamSet * matchEnd = mPB.CreateStreamSet(1);
        mPB.CreateKernelFamilyCall<RE_Kernel>(mCtxt, re, matchEnd);
        auto minlgth = getLengthRange(re, mCtxt.mLengthAlphabet).first;
        auto offset = grepOffset(re);
        mPB.CreateKernelFamilyCall<FixedMatchSpansKernel>(minlgth, offset, matchEnd, spans);
        if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
            auto spanName = "minlen" + std::to_string(minlgth + offset);
            mPB.captureBitstream(spanName, spans);
        }
    }
}

RE * RE_PipelineBuilder::spanFactoring(RE * re) {
    re::FixedSpanNamer FLnamer(mCtxt.mCodeUnitAlphabet);
    RE * xfrmedRE = FLnamer.transformRE(re);
    xfrmedRE = mUPnamer.transformRE(xfrmedRE);
    //re::Repeated_CC_Seq_Namer RCCSnamer;
    //xfrmedRE = mRCCSnamer.transformRE(xfrmedRE);
    return xfrmedRE;
}

RE * RE_PipelineBuilder::prepareRE(RE * re) {
    const cc::Alphabet * lengthAlphabet = mCtxt.mLengthAlphabet;
    RE * xfrmedRE = expandPermutes(re);
    xfrmedRE = regular_expression_passes(xfrmedRE);

    UCD::PropertyExternalizer PE;
    xfrmedRE = PE.transformRE(xfrmedRE);
    for (auto m : PE.mNameMap) {
        if (re::PropertyExpression * pe = dyn_cast<re::PropertyExpression>(m.second)) {
            compileProperty(pe);
        }
    }

    re::LookAheadNamer LA;
    xfrmedRE = LA.transformRE(xfrmedRE);

    re::VariableLengthCCNamer CCnamer;
    xfrmedRE = CCnamer.transformRE(xfrmedRE);
    for (auto m : CCnamer.mNameMap) {
        std::vector<re::CC *> ccs = {cast<re::CC>(m.second)};
        StreamSet * ccStrm = mPB.CreateStreamSet(1);
        mPB.CreateKernelFamilyCall<CharClassesKernel>(ccs, mCtxt.mCodeUnitStream, ccStrm);
        addExternal(m.first, ExternalStream{ExternalStreamKind::FixedLength, 0, {1, 1}, ccStrm});
    }

    if (mMatchSpans) {
        xfrmedRE = zeroBoundElimination(xfrmedRE);
        xfrmedRE = variableAltPromotion(xfrmedRE, lengthAlphabet);
    }

    if (mCtxt.mCodeUnitAlphabet == &cc::UTF8) {
        xfrmedRE = toUTF8(xfrmedRE);
    }
    return xfrmedRE;
}

RE * RE_PipelineBuilder::processReferences(RE * re) {
    re::ReferenceInfo mRefInfo = re::buildReferenceInfo(re);
    if (!mRefInfo.twixtREs.empty()) {
        re::FixedReferenceTransformer FRT(mRefInfo);
        RE * xfrmed = FRT.transformRE(re);
        for (auto m : FRT.mNameMap) {
            auto name = m.first;
            re::Reference * ref = cast<re::Reference>(m.second);
            UCD::property_t p = ref->getReferencedProperty();
            std::string instanceName = ref->getInstanceName();
            auto captureLen = getLengthRange(ref->getCapture(), &cc::Unicode).first;
            if (captureLen != 1) {
                llvm::report_fatal_error("Capture length > 1 is a future extension");
            }
            auto mapping = mRefInfo.twixtREs.find(instanceName);
            auto twixtLen = getLengthRange(mapping->second, &cc::Unicode).first;
            auto dist = captureLen + twixtLen;
            UCD::PropertyObject * propObj = UCD::getPropertyObject(p);
            if (auto * obj = dyn_cast<UCD::EnumeratedPropertyObject>(propObj)) {
                std::string extName = UCD::getPropertyFullName(p) + "_basis";
                std::vector<UCD::UnicodeSet> & bases = obj->GetEnumerationBasisSets();
                std::vector<re::CC *> ccs;
                for (auto & b : bases) ccs.push_back(makeCC(b, &cc::Unicode));
                StreamSet * propertyBasis = mPB.CreateStreamSet(ccs.size());
                mPB.CreateKernelFamilyCall<CharClassesKernel>(ccs, mCtxt.mCodeUnitStream, propertyBasis);
                StreamSet * distStrm = mPB.CreateStreamSet(1);
                mPB.CreateKernelCall<FixedDistanceMatchesKernel>(dist, propertyBasis, distStrm);
                addExternal(name, ExternalStream{ExternalStreamKind::FixedLength, 0u, {1, 1}, distStrm});
            } else if (isa<UCD::CodePointPropertyObject>(propObj)) {
                // Identity or other codepoint properties
                StreamSet * distStrm = mPB.CreateStreamSet(1);
                mPB.CreateKernelCall<CodePointMatchKernel>(p, dist, mCtxt.mCodeUnitStream, distStrm);
                addExternal(name, ExternalStream{ExternalStreamKind::FixedLength, 0u, {1, 1}, distStrm});
            } else {
                llvm::report_fatal_error("Property reference must be an enumerated or codepoint property.");
            }
        }
        return xfrmed;
    }
    return re;
}


void UnicodePropertyLogic(PipelineBuilder & P, PropertyExpression * pe,
                          StreamSet * BasisBits, StreamSet * PropertyStream) {
    UnicodePropertyLogic(P, pe, BasisBits, nullptr, PropertyStream);
}

void UnicodePropertyLogic(PipelineBuilder & P, re::PropertyExpression * pe,
                          StreamSet * BasisBits, StreamSet * IndexStream, StreamSet * PropertyStream) {
    //pe = cast<PropertyExpression>(UCD::linkAndResolve(pe, grep::lineNumGrep));
    std::string propName = pe->getFullName();
    if (pe->getKind() == re::PropertyExpression::Kind::Codepoint) {
        P.CreateKernelFamilyCall<UnicodePropertyKernelBuilder>(pe, BasisBits, PropertyStream);
    } else { //PropertyExpression::Kind::Boundary
        if (BasisBits->getNumElements() < 21) {
            if (IndexStream == nullptr) {
                llvm::report_fatal_error("index stream required for boundary properties without full Unicode basis");
            }
        }
        UCD::property_t prop = static_cast<UCD::property_t>(pe->getPropertyCode());
        UCD::PropertyObject * propObj = UCD::getPropertyObject(prop);
        if (UCD::BoundaryPropertyObject * bObj = dyn_cast<UCD::BoundaryPropertyObject>(propObj)) {
            // Grapheme Cluster and level 2 word boundaries \b{g}, \b{w} 
            re::RE * bRE = bObj->GetBoundaryExpression();
            const auto b_Sets = re::collectCCs(bRE, cc::Unicode, re::NameProcessingMode::ProcessDefinition);
            auto b_mpx = cc::makeMultiplexedAlphabet("b_mpx", b_Sets);
            bRE = transformCCs(b_mpx, bRE, re::NameTransformationMode::TransformDefinition);
            auto b_basis = b_mpx->getMultiplexedCCs();
            StreamSet * b_Classes = P.CreateStreamSet(b_basis.size());
            P.CreateKernelFamilyCall<CharClassesKernel>(b_basis, BasisBits, b_Classes);
            re::LookAheadNamer LA;
            bRE = LA.transformRE(bRE);
            //
            RE_CompilerContext ctxt;

            if (IndexStream && (prop == UCD::w)) {
                // Must switch to Unicode indexing for word boundaries.
                StreamSet * b_Classes_Uindexed = P.CreateStreamSet(b_basis.size());
                FilterByMask(P, IndexStream, b_Classes, b_Classes_Uindexed);
                ctxt.setCodeUnitContext(b_mpx, b_Classes_Uindexed);
                StreamSet * U_property = P.CreateStreamSet(1);
                RE_PipelineBuilder RE_PB(P, ctxt);
                RE_PB.matchSearchPipeline(bRE, U_property);
                SpreadByMask(P, IndexStream, U_property, PropertyStream);
            } else {
                if (BasisBits->getNumElements() < 21) {
                    ctxt.setCodeUnitContext(&cc::UTF8, BasisBits);
                } else {
                    ctxt.setCodeUnitContext(&cc::Unicode, BasisBits);
                }
                ctxt.addAlphabet(b_mpx, b_Classes);
                if (IndexStream) {
                    ctxt.setIndexingContext(&cc::Unicode, IndexStream);
                }
                RE_PipelineBuilder RE_PB(P, ctxt);
                RE_PB.matchSearchPipeline(bRE, PropertyStream);
            }
        } else if (auto * obj = dyn_cast<UCD::EnumeratedPropertyObject>(propObj)) {
            std::vector<UCD::UnicodeSet> & bases = obj->GetEnumerationBasisSets();
            std::vector<re::CC *> ccs;
            for (auto & b : bases) ccs.push_back(makeCC(b, &cc::Unicode));
            StreamSet * enumBasis = P.CreateStreamSet(ccs.size());
            P.CreateKernelFamilyCall<CharClassesKernel>(ccs, BasisBits, enumBasis);
            P.CreateKernelCall<BoundaryKernel>(enumBasis, IndexStream, PropertyStream);
        } else if (auto * obj = dyn_cast<UCD::BinaryPropertyObject>(propObj)) {
            std::vector<re::CC *> ccs = {makeCC(obj->GetCodepointSet("Y"), &cc::Unicode)};
            StreamSet * pStrm = P.CreateStreamSet(1);
            P.CreateKernelFamilyCall<CharClassesKernel>(ccs, BasisBits, pStrm);
            P.CreateKernelCall<BoundaryKernel>(pStrm, IndexStream, PropertyStream);
        } else {
            llvm::report_fatal_error("Unsupported property for boundary expression");
        }
    }
}
