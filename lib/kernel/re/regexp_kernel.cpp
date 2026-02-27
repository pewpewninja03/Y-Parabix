#include <kernel/re/regexp_kernel.h>
#include <kernel/core/kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/bitwise/bixlogic.h>
#include <kernel/streamutils/stream_shift.h>
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
#include <re/transforms/regex_passes.h>
#include <re/transforms/to_utf8.h>
#include <re/transforms/re_multiplex.h>
#include <re/transforms/expand_permutes.h>
#include <re/transforms/name_intro.h>
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
        //llvm::errs() << "RE_Kernel - adding external: " << name << "\n";
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
        bool fromFirst = false;
        if (ext.kind == ExternalStreamKind::StartIndexed) {
            fromFirst = true;
        }
        re_compiler.addPrecompiled(name, RE_Compiler::ExternalStream(RE_Compiler::Marker(extStrm, offset), ext.lgthRange, fromFirst));
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


void RE_PipelineBuilder::compileProperty(PropertyExpression * pe) {
    std::string propName = pe->getFullName();
    if (pe->getKind() == re::PropertyExpression::Kind::Codepoint) {
        StreamSet * pStrm  = mPB.CreateStreamSet(1);
        mPB.CreateKernelFamilyCall<UnicodePropertyKernelBuilder>(pe, mCtxt.mCodeUnitStream, pStrm);
        mCtxt.mExternals.emplace(propName, ExternalStream{ExternalStreamKind::FixedLength, 0, {1, 1}, pStrm});
        if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
            mPB.captureBitstream(propName, pStrm);
        }
    } else { //PropertyExpression::Kind::Boundary
        UCD::property_t prop = static_cast<UCD::property_t>(pe->getPropertyCode());
        if (prop == UCD::g) {
            re::RE * GCB_RE = re::generateGraphemeClusterBoundaryRule();
            const auto GCB_Sets = re::collectCCs(GCB_RE, cc::Unicode, re::NameProcessingMode::ProcessDefinition);
            auto GCB_mpx = cc::makeMultiplexedAlphabet("GCB_mpx", GCB_Sets);
            GCB_RE = transformCCs(GCB_mpx, GCB_RE, re::NameTransformationMode::TransformDefinition);
            auto GCB_basis = GCB_mpx->getMultiplexedCCs();
            StreamSet * const GCB_Classes = mPB.CreateStreamSet(GCB_basis.size());
            mPB.CreateKernelFamilyCall<CharClassesKernel>(GCB_basis, mCtxt.mCodeUnitStream, GCB_Classes);
            // TO DO:  Consider setting up a local RE_CompilerContext 
            mCtxt.addAlphabet(GCB_mpx, GCB_Classes);
            StreamSet * GCB_strm = mPB.CreateStreamSet(1);
            mPB.CreateKernelFamilyCall<RE_Kernel>(mCtxt, GCB_RE, GCB_strm);
            mCtxt.mExternals.emplace("\\b{g}", ExternalStream{ExternalStreamKind::ZeroWidth, 1, {0, 0}, GCB_strm});
            if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
                mPB.captureBitstream("\\b{g}", GCB_strm);
            }
        } else if (prop == UCD::w) {
            re::RE * WB_RE = re::generateWordBoundaryRule();
            const auto WB_Sets = re::collectCCs(WB_RE, cc::Unicode, re::NameProcessingMode::ProcessDefinition);
            auto WB_mpx = cc::makeMultiplexedAlphabet("WB_mpx", WB_Sets);
            WB_RE = transformCCs(WB_mpx, WB_RE, re::NameTransformationMode::TransformDefinition);
            auto WB_basis = WB_mpx->getMultiplexedCCs();
            StreamSet * const WB_Classes = mPB.CreateStreamSet(WB_basis.size());
            // TO DO:  Consider setting up a local RE_CompilerContext 
            mCtxt.addAlphabet(WB_mpx, WB_Classes);
            mPB.CreateKernelFamilyCall<CharClassesKernel>(WB_basis, mCtxt.mCodeUnitStream, WB_Classes);
            // TODO: Deal with lookahead 2 ???  --- use a RE_Pipeline call??
            StreamSet * WB_strm = mPB.CreateStreamSet(1);
            mPB.CreateKernelFamilyCall<RE_Kernel>(mCtxt, WB_RE, WB_strm);
            mCtxt.mExternals.emplace("\\b{w}", ExternalStream{ExternalStreamKind::ZeroWidth, 1, {0, 0}, WB_strm});
            if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
                mPB.captureBitstream("\\b{w}", WB_strm);
            }
        } else {  // Boundary expressions, except GCB, level 2 WB.
            UCD::PropertyObject * propObj = UCD::getPropertyObject(prop);
            if (auto * obj = dyn_cast<UCD::EnumeratedPropertyObject>(propObj)) {
                std::vector<UCD::UnicodeSet> & bases = obj->GetEnumerationBasisSets();
                std::vector<re::CC *> ccs;
                for (auto & b : bases) ccs.push_back(makeCC(b, &cc::Unicode));
                StreamSet * basis = mPB.CreateStreamSet(ccs.size());
                mPB.CreateKernelFamilyCall<CharClassesKernel>(ccs, mCtxt.mCodeUnitStream, basis);
                StreamSet * bStrm  = mPB.CreateStreamSet(1);
                mPB.CreateKernelCall<BoundaryKernel>(basis, mCtxt.mIndexStream, bStrm);
                mCtxt.mExternals.emplace(propName, ExternalStream{ExternalStreamKind::ZeroWidth, 1, {0, 0}, bStrm});
                if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
                    mPB.captureBitstream(propName, bStrm);
                }
            } else if (auto * obj = dyn_cast<UCD::BinaryPropertyObject>(propObj)) {
                std::vector<re::CC *> ccs = {makeCC(obj->GetCodepointSet("Y"), &cc::Unicode)};
                StreamSet * pStrm = mPB.CreateStreamSet(1);
                mPB.CreateKernelFamilyCall<CharClassesKernel>(ccs, mCtxt.mCodeUnitStream, pStrm);
                StreamSet * bStrm  = mPB.CreateStreamSet(1);
                mPB.CreateKernelCall<BoundaryKernel>(pStrm, mCtxt.mIndexStream, bStrm);
                mCtxt.mExternals.emplace(propName, ExternalStream{ExternalStreamKind::ZeroWidth, 1, {0, 0}, bStrm});
                if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
                    mPB.captureBitstream(propName, bStrm);
                }
            } else {
                llvm::report_fatal_error("Expected enumerated property for boundary expression");
            }
        }
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
    unsigned amt = NamedLookAheadAmount(n, *mCtxt.mCodeUnitAlphabet);
    unsigned offset = grepOffset(defn);
    if (amt == 0) {
        // Not a lookahead; compile using a single kernel call.
        StreamSet * extStrm = mPB.CreateStreamSet(1);
        mPB.CreateKernelFamilyCall<RE_Kernel>(mCtxt, defn, extStrm);
        auto r = getLengthRange(defn, mCtxt.mCodeUnitAlphabet);
        if (r.second == 0) {
            mCtxt.mExternals.emplace(name, ExternalStream{ExternalStreamKind::ZeroWidth, 1, r, extStrm});
        } else if (r.first == r.second) {
            mCtxt.mExternals.emplace(name, ExternalStream{ExternalStreamKind::FixedLength, offset, r, extStrm});
        } else {
            mCtxt.mExternals.emplace(name, ExternalStream{ExternalStreamKind::EndIndexed, offset, r, extStrm});
        }
    } else {
        RE * asserted = llvm::cast<Assertion>(defn)->getAsserted();
        StreamSet * assertedStrm = mPB.CreateStreamSet(1);
        mPB.CreateKernelFamilyCall<RE_Kernel>(mCtxt, asserted, assertedStrm);
        auto r = getLengthRange(asserted, mCtxt.mCodeUnitAlphabet);
        if (r.first == r.second) {
            // Fixed length lookaheads can be stored directly.
            mCtxt.mExternals.emplace(name, ExternalStream{ExternalStreamKind::StartIndexed, r.second, r, assertedStrm});
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
            mCtxt.mExternals.emplace(name, ExternalStream{ExternalStreamKind::StartIndexed, amt, r, extStrm});
        }
    }
}

void RE_PipelineBuilder::createRE_Pipeline(RE * re, StreamSet * results) {
    RE * xfrmdRE = prepareRE(re);
    //llvm::errs() << "After prepareRE:" << Printer_RE::PrintRE(xfrmdRE) << "\n";
    xfrmdRE = processReferences(xfrmdRE);
    prepareExternals(xfrmdRE);
    mPB.CreateKernelFamilyCall<RE_Kernel>(mCtxt, xfrmdRE, results);
}

RE * RE_PipelineBuilder::prepareRE(RE * re) {
    const cc::Alphabet * lengthAlphabet = mCtxt.mIndexingAlphabet ? mCtxt.mIndexingAlphabet : mCtxt.mCodeUnitAlphabet;
    RE * xfrmedRE = expandPermutes(re);
    xfrmedRE = regular_expression_passes(xfrmedRE);

    UCD::PropertyExternalizer PE;
    xfrmedRE = PE.transformRE(xfrmedRE);
    for (auto m : PE.mNameMap) {
        if (re::PropertyExpression * pe = dyn_cast<re::PropertyExpression>(m.second)) {
            compileProperty(pe);
        }
    }

    re::VariableLengthCCNamer CCnamer;
    xfrmedRE = CCnamer.transformRE(xfrmedRE);
    for (auto m : CCnamer.mNameMap) {
        std::vector<re::CC *> ccs = {cast<re::CC>(m.second)};
        StreamSet * ccStrm = mPB.CreateStreamSet(1);
        mPB.CreateKernelFamilyCall<CharClassesKernel>(ccs, mCtxt.mCodeUnitStream, ccStrm);
        mCtxt.mExternals.emplace(m.first, ExternalStream{ExternalStreamKind::FixedLength, 0, {1, 1}, ccStrm});
    }

    if (mMatchSpans) {
        xfrmedRE = zeroBoundElimination(xfrmedRE);
        xfrmedRE = variableAltPromotion(xfrmedRE, lengthAlphabet);
    }

    if (mCtxt.mCodeUnitAlphabet == &cc::UTF8) {
        if (mCtxt.mIndexingAlphabet == nullptr) {
            xfrmedRE = toUTF8(xfrmedRE);
        } else {
            xfrmedRE = toUTF8(xfrmedRE, /* useInternalNaming = */ true);
        }
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
                mCtxt.mExternals.emplace(name, ExternalStream{ExternalStreamKind::FixedLength, 0u, {1, 1}, distStrm});
            } else if (isa<UCD::CodePointPropertyObject>(propObj)) {
                // Identity or other codepoint properties
                StreamSet * distStrm = mPB.CreateStreamSet(1);
                mPB.CreateKernelCall<CodePointMatchKernel>(p, dist, mCtxt.mCodeUnitStream, distStrm);
                mCtxt.mExternals.emplace(name, ExternalStream{ExternalStreamKind::FixedLength, 0u, {1, 1}, distStrm});
            } else {
                llvm::report_fatal_error("Property reference must be an enumerated or codepoint property.");
            }
        }
        return xfrmed;
    }
    return re;
}

