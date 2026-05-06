/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <grep/grep_kernel.h>

#include <grep/grep_engine.h>
#include <grep/grep_toolchain.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/codegenstate.h>
#include <toolchain/toolchain.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>          // for Ones
#include <pablo/pe_var.h>           // for Var
#include <pablo/pe_zeroes.h>        // for Zeroes
#include <pablo/pe_infile.h>
#include <pablo/pe_advance.h>
#include <pablo/boolean.h>
#include <pablo/pe_count.h>
#include <pablo/pe_matchstar.h>
#include <pablo/pe_pack.h>
#include <pablo/pe_debugprint.h>
#include <re/printer/re_printer.h>
#include <re/adt/re_cc.h>
#include <re/adt/re_name.h>
#include <re/alphabet/alphabet.h>
#include <re/analysis/re_analysis.h>
#include <re/toolchain/toolchain.h>
#include <re/transforms/re_reverse.h>
#include <re/transforms/re_transformer.h>
#include <re/transforms/to_utf8.h>
#include <re/analysis/collect_ccs.h>
#include <re/transforms/exclude_CC.h>
#include <re/transforms/re_multiplex.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/re/regexp_kernel.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/streamutils/streams_merge.h>
#include <kernel/unicode/boundary_kernels.h>
#include <kernel/unicode/utf8_decoder.h>
#include <kernel/unicode/utf8_support.h>
#include <kernel/unicode/UCD_property_kernel.h>
#include <re/analysis/re_name_gather.h>
#include <re/unicode/boundaries.h>
#include <re/unicode/re_name_resolve.h>
#include <re/unicode/resolve_properties.h>
#include <kernel/unicode/charclasses.h>
#include <re/cc/cc_compiler.h>         // for CC_Compiler
#include <re/cc/cc_compiler_target.h>
#include <re/cc/cc_kernel.h>
#include <re/alphabet/multiplex_CCs.h>
#include <re/compile/re_compiler.h>
#include <ucd/data/PropertyAliases.h>
#include <ucd/data/PropertyObjectTable.h>
#include <ucd/utf/utf_compiler.h>

using namespace kernel;
using namespace pablo;
using namespace re;
using namespace llvm;



unsigned round_up_to_blocksize(int lgth) {
    unsigned lookahead_blocks = (codegen::BlockSize - 1 + lgth)/codegen::BlockSize;
    return lookahead_blocks * codegen::BlockSize;
}


void MatchedLinesKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    auto matchResults = getInputStreamSet("matchResults");
    PabloAST * lineBreaks = pb.createExtract(getInputStreamVar("lineBreaks"), pb.getInteger(0));
    PabloAST * notLB = pb.createNot(lineBreaks);
    PabloAST * match_follow = pb.createMatchStar(matchResults.back(), notLB);
    Var * const matchedLines = getOutputStreamVar("matchedLines");
    pb.createAssign(pb.createExtract(matchedLines, pb.getInteger(0)), pb.createAnd(match_follow, lineBreaks, "matchedLines"));
}

MatchedLinesKernel::MatchedLinesKernel (LLVMTypeSystemInterface & ts, StreamSet * Matches, StreamSet * LineBreakStream, StreamSet * MatchedLines)
: PabloKernel(ts, "MatchedLines" + std::to_string(Matches->getNumElements()),
// inputs
{Binding{"matchResults", Matches}
,Binding{"lineBreaks", LineBreakStream, FixedRate(), Principal()}},
// output
{Binding{"matchedLines", MatchedLines}}) {

}

void InvertMatchesKernel::generateDoBlockMethod(KernelBuilder & b) {
    Value * input = b.loadInputStreamBlock("matchedLines", b.getInt32(0));
    Value * lbs = b.loadInputStreamBlock("lineBreaks", b.getInt32(0));
    Value * inverted = b.CreateAnd(b.CreateNot(input), lbs, "inverted");
    b.storeOutputStreamBlock("nonMatches", b.getInt32(0), inverted);
}

InvertMatchesKernel::InvertMatchesKernel(LLVMTypeSystemInterface & ts, StreamSet * Matches, StreamSet * LineBreakStream, StreamSet * InvertedMatches)
: BlockOrientedKernel(ts, "InvertMatches" + std::to_string(Matches->getNumElements()),
// Inputs
{Binding{"matchedLines", Matches},
 Binding{"lineBreaks", LineBreakStream}},
// Outputs
{Binding{"nonMatches", InvertedMatches}},
// Input/Output Scalars and internal state
{}, {}, {}) {

}

SpansToMarksKernel::SpansToMarksKernel(LLVMTypeSystemInterface & ts, StreamSet * Spans, StreamSet * Marks)
: PabloKernel(ts, "SpansToMarksKernel",
{Binding{"Spans", Spans}}, {Binding{"Marks", Marks}}) {}

void SpansToMarksKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * spans = getInputStreamSet("Spans")[0];
    Var * matchEndsVar = getOutputStreamVar("Marks");
    PabloAST * starts = pb.createAnd(spans, pb.createNot(pb.createAdvance(spans, 1)), "starts");
    PabloAST * follows = pb.createAnd(pb.createAdvance(spans, 1), pb.createNot(spans), "follows");
    pb.createAssign(pb.createExtract(matchEndsVar, 0), starts);
    pb.createAssign(pb.createExtract(matchEndsVar, 1), follows);
}

void PopcountKernel::generatePabloMethod() {
    auto pb = getEntryScope();
    const auto toCount = pb->createExtract(getInputStreamVar("toCount"), pb->getInteger(0));
    const auto countResult = getOutputScalarVar("countResult");
    const auto newCount = pb->createCount(pb->createInFile(toCount));
    pb->createAssign(countResult, newCount);
}

PopcountKernel::PopcountKernel (LLVMTypeSystemInterface & ts, StreamSet * const toCount, Scalar * countResult)
: PabloKernel(ts, "Popcount",
{Binding{"toCount", toCount}},
{},
{},
{Binding{"countResult", countResult}}) {

}

void AbortOnNull::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    Module * const m = b.getModule();
    DataLayout DL(m);
    IntegerType * const intPtrTy = DL.getIntPtrType(m->getContext());
    Type * voidPtrTy = b.getVoidPtrTy();
    Type * blockTy = b.getBitBlockType();
    const auto blocksPerStride = getStride() / b.getBitBlockWidth();
    Constant * const BLOCKS_PER_STRIDE = b.getSize(blocksPerStride);
    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const strideLoop = b.CreateBasicBlock("strideLoop");
    BasicBlock * const stridesDone = b.CreateBasicBlock("stridesDone");
    BasicBlock * const nullByteDetection = b.CreateBasicBlock("nullByteDetection");
    BasicBlock * const nullByteFound = b.CreateBasicBlock("nullByteFound");
    BasicBlock * const finalStride = b.CreateBasicBlock("finalStride");
    BasicBlock * const segmentDone = b.CreateBasicBlock("segmentDone");

    Value * const numOfBlocks = b.CreateMul(numOfStrides, BLOCKS_PER_STRIDE);
    Value * itemsToDo = b.getAccessibleItemCount("byteData");
    //
    // Fast loop to prove that there are no null bytes in a multiblock region.
    // We repeatedly combine byte packs using a SIMD unsigned min operation
    // (implemented as a Select/ICmpULT combination).
    //
    Value * byteStreamBasePtr = b.getInputStreamBlockPtr("byteData", b.getSize(0), b.getSize(0));
    Value * outputStreamBasePtr = b.getOutputStreamBlockPtr("untilNull", b.getSize(0), b.getSize(0));

    //
    // We set up a a set of eight accumulators to accumulate the minimum byte
    // values seen at each position in a block.   The initial min value at
    // each position is 0xFF (all ones).
    Value * blockMin[8];
    for (unsigned i = 0; i < 8; i++) {
        blockMin[i] = b.fwCast(8, b.allOnes());
    }
    // If we're in the final block bypass the fast loop.
    b.CreateCondBr(b.isFinal(), finalStride, strideLoop);

    b.SetInsertPoint(strideLoop);
    PHINode * const baseBlockIndex = b.CreatePHI(b.getSizeTy(), 2);
    baseBlockIndex->addIncoming(ConstantInt::get(baseBlockIndex->getType(), 0), entry);
    PHINode * const blocksRemaining = b.CreatePHI(b.getSizeTy(), 2);
    blocksRemaining->addIncoming(numOfBlocks, entry);
    FixedArray<Value *, 2> indices;
    indices[0] = baseBlockIndex;
    for (unsigned i = 0; i < 8; i++) {
        indices[1] = b.getSize(i);
        Value * next = b.CreateBlockAlignedLoad(blockTy, b.CreateGEP(blockTy, byteStreamBasePtr, indices));
        b.CreateBlockAlignedStore(next, b.CreateGEP(blockTy, outputStreamBasePtr, indices));
        next = b.fwCast(8, next);
        blockMin[i] = b.CreateSelect(b.CreateICmpULT(next, blockMin[i]), next, blockMin[i]);
    }
    Value * nextBlockIndex = b.CreateAdd(baseBlockIndex, ConstantInt::get(baseBlockIndex->getType(), 1));
    Value * nextRemaining = b.CreateSub(blocksRemaining, ConstantInt::get(blocksRemaining->getType(), 1));
    baseBlockIndex->addIncoming(nextBlockIndex, strideLoop);
    blocksRemaining->addIncoming(nextRemaining, strideLoop);
    b.CreateCondBr(b.CreateICmpUGT(nextRemaining, ConstantInt::getNullValue(blocksRemaining->getType())), strideLoop, stridesDone);

    b.SetInsertPoint(stridesDone);
    // Combine the 8 blockMin values.
    for (unsigned i = 0; i < 4; i++) {
        blockMin[i] = b.CreateSelect(b.CreateICmpULT(blockMin[i], blockMin[i+4]), blockMin[i], blockMin[i+4]);
    }
    for (unsigned i = 0; i < 2; i++) {
        blockMin[i] = b.CreateSelect(b.CreateICmpULT(blockMin[i], blockMin[i+4]), blockMin[i], blockMin[i+2]);
    }
    blockMin[0] = b.CreateSelect(b.CreateICmpULT(blockMin[0], blockMin[1]), blockMin[0], blockMin[1]);
    Value * anyNull = b.bitblock_any(b.simd_eq(8, blockMin[0], b.allZeroes()));

    b.CreateCondBr(anyNull, nullByteDetection, segmentDone);


    b.SetInsertPoint(finalStride);
    b.CreateMemCpy(b.CreatePointerCast(outputStreamBasePtr, voidPtrTy), b.CreatePointerCast(byteStreamBasePtr, voidPtrTy), itemsToDo, 1);
    b.CreateBr(nullByteDetection);

    b.SetInsertPoint(nullByteDetection);
    //  Find the exact location using memchr, which should be fast enough.
    //
    Value * ptrToNull = b.CreateMemChr(b.CreatePointerCast(byteStreamBasePtr, voidPtrTy), b.getInt32(0), itemsToDo);
    Value * ptrAddr = b.CreatePtrToInt(ptrToNull, intPtrTy);
    b.CreateCondBr(b.CreateICmpEQ(ptrAddr, ConstantInt::getNullValue(intPtrTy)), segmentDone, nullByteFound);

    // A null byte has been located; set the termination code and call the signal handler.
    b.SetInsertPoint(nullByteFound);
    Value * nullPosn = b.CreateSub(b.CreatePtrToInt(ptrToNull, intPtrTy), b.CreatePtrToInt(byteStreamBasePtr, intPtrTy));
    b.setFatalTerminationSignal();
    Function * const dispatcher = m->getFunction("signal_dispatcher"); assert (dispatcher);
    Value * handler = b.getScalarField("handler_address");
    b.CreateCall(dispatcher, {handler, ConstantInt::get(b.getInt32Ty(), static_cast<unsigned>(grep::GrepSignal::BinaryFile))});
    b.CreateBr(segmentDone);

    b.SetInsertPoint(segmentDone);
    PHINode * const produced = b.CreatePHI(b.getSizeTy(), 3);
    produced->addIncoming(nullPosn, nullByteFound);
    produced->addIncoming(itemsToDo, stridesDone);
    produced->addIncoming(itemsToDo, nullByteDetection);
    Value * producedCount = b.getProducedItemCount("untilNull");
    producedCount = b.CreateAdd(producedCount, produced);
    b.setProducedItemCount("untilNull", producedCount);
}

AbortOnNull::AbortOnNull(LLVMTypeSystemInterface & ts, StreamSet * const InputStream, StreamSet * const OutputStream, Scalar * callbackObject)
: MultiBlockKernel(ts, "AbortOnNull",
// inputs
{Binding{"byteData", InputStream, FixedRate(), Principal()}},
// outputs
{Binding{ "untilNull", OutputStream, FixedRate(), Deferred()}},
// input scalars
{Binding{"handler_address", callbackObject}},
{}, {}) {
    addAttribute(CanTerminateEarly());
    addAttribute(MayFatallyTerminate());
}

ContextSpan::ContextSpan(LLVMTypeSystemInterface & ts, StreamSet * const markerStream, StreamSet * const contextStream, unsigned before, unsigned after)
: PabloKernel(ts, "ContextSpan-" + std::to_string(before) + "+" + std::to_string(after),
              // input
{Binding{"markerStream", markerStream, FixedRate(1), LookAhead(before)}},
              // output
{Binding{"contextStream", contextStream}}),
mBeforeContext(before), mAfterContext(after) {
}

void ContextSpan::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    Var * markerStream = pb.createExtract(getInputStreamVar("markerStream"), pb.getInteger(0));
    PabloAST * contextStart = pb.createLookahead(markerStream, pb.getInteger(mBeforeContext));
    unsigned lgth = mBeforeContext + 1 + mAfterContext;
    PabloAST * consecutive = contextStart;
    unsigned consecutiveCount = 1;
    for (unsigned i = 1; i <= lgth/2; i *= 2) {
        consecutiveCount += i;
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, i),
                                  "consecutive" + std::to_string(consecutiveCount));
    }
    if (consecutiveCount < lgth) {
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, lgth - consecutiveCount),
                                  "consecutive" + std::to_string(lgth));
    }
    pb.createAssign(pb.createExtract(getOutputStreamVar("contextStream"), pb.getInteger(0)), pb.createInFile(consecutive));
}
LongestMatchMarks::LongestMatchMarks(LLVMTypeSystemInterface & ts, StreamSet * start_ends, StreamSet * marks)
: PabloKernel(ts, "LongestMatchMarks"  + std::to_string(marks->getNumElements()) + "x1",
              {Binding{"start_ends", start_ends, FixedRate(1), LookAhead(1)}},
              {Binding{"marks", marks}}) {}

void LongestMatchMarks::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> starts_ends = getInputStreamSet("start_ends");
    PabloAST * starts;
    PabloAST * ends;
    if (starts_ends.size() == 2) {
        starts = starts_ends[0];
        ends = starts_ends[1];
    } else {
        starts = pb.createNot(starts_ends[0]);
        ends = starts_ends[0];
    }
    PabloAST * end_follows = pb.createLookahead(ends, 1);
    PabloAST * span_starts = pb.createAnd(starts, end_follows, "span_starts");
    PabloAST * span_ends = pb.createAnd(ends, pb.createNot(end_follows), "span_ends");
    Var * marksVar = getOutputStreamVar("marks");
    pb.createAssign(pb.createExtract(marksVar, pb.getInteger(0)), span_starts);
    pb.createAssign(pb.createExtract(marksVar, pb.getInteger(1)), span_ends);
}

unsigned spanLookAhead(unsigned offset1, unsigned offset2) {
    return round_up_to_blocksize(std::max(offset1, offset2));
}

InclusiveSpans::InclusiveSpans(LLVMTypeSystemInterface & ts,
                               unsigned prefixOffset, unsigned suffixOffset,
                               StreamSet * marks, StreamSet * spans)
: PabloKernel(ts, "InclusiveSpans@" + std::to_string(prefixOffset) + ":" + std::to_string(suffixOffset),
              {Binding{"marks", marks, FixedRate(1),
                                       LookAhead(spanLookAhead(prefixOffset, suffixOffset))}},
              {Binding{"spans", spans}}),
    mPrefixOffset(prefixOffset), mSuffixOffset(suffixOffset) {
}

void InclusiveSpans::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    Var * marksVar = getInputStreamVar("marks");
    PabloAST * starts = pb.createExtract(marksVar, pb.getInteger(0));
    if (mPrefixOffset > 0) {
        starts = pb.createLookahead(starts, mPrefixOffset);
    }
    PabloAST * ends = pb.createExtract(marksVar, pb.getInteger(1));
    if (mSuffixOffset > 0) {
        ends = pb.createLookahead(ends, mSuffixOffset);
    }
    PabloAST * spans = pb.createIntrinsicCall(pablo::Intrinsic::InclusiveSpan, {starts, ends});
    pb.createAssign(pb.createExtract(getOutputStreamVar("spans"), pb.getInteger(0)), spans);
}

std::string CC_string(std::vector<CC *> transitionCCs, StreamSet * index) {
    std::stringstream s;
    if (index != nullptr) s << "+ix";
    for (auto & cc : transitionCCs) {
        s << "_" << cc->canonicalName();
    }
    return s.str();
}

MaskCC::MaskCC(LLVMTypeSystemInterface & ts, CC * CC_to_mask, StreamSet * basis, StreamSet * mask, StreamSet * index)
: PabloKernel(ts, "MaskCC" + basis->shapeString()
                 + CC_string(std::vector<CC *>{CC_to_mask}, index)
                 + UTF::kernelAnnotation(),
              {Binding{"basis", basis}},
              {Binding{"mask", mask}}), mCC_to_mask(CC_to_mask), mIndexStrm(nullptr) {
                  if (index != nullptr) {
                      mInputStreamSets.push_back(Binding{"index", index});
                      mIndexStrm = index;
                  }
              }

void MaskCC::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
    Var * maskVar = pb.createVar("maskVar", pb.createZeroes());
    unicodeCompiler.compile({maskVar}, {mCC_to_mask});
    PabloAST * mask = pb.createNot(maskVar);
    if (mIndexStrm) {
        PabloAST * idx = getInputStreamSet("index")[0];
        mask = pb.createAnd(idx, mask);
    }
    pb.createAssign(pb.createExtract(getOutputStreamVar("mask"), pb.getInteger(0)), mask);
}

MaskSelfTransitions::MaskSelfTransitions(LLVMTypeSystemInterface & ts, const std::vector<CC *> transitionCCs, StreamSet * basis, StreamSet * mask, StreamSet * index)
: PabloKernel(ts, "MaskSelfTransitions" + basis->shapeString() + CC_string(transitionCCs, index),
              {Binding{"basis", basis}},
              {Binding{"mask", mask}}), mTransitionCCs(transitionCCs), mIndexStrm(nullptr) {
                  if (index != nullptr) {
                      mInputStreamSets.push_back(Binding{"index", index});
                      mIndexStrm = index;
                  }
              }

void MaskSelfTransitions::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    std::unique_ptr<cc::CC_Compiler> ccc;
    if (basis.size() == 1) {
        ccc = std::make_unique<cc::Direct_CC_Compiler>(basis[0]);
    } else {
        ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(basis);
    }
    PabloAST * transitions = pb.createZeroes();
    PabloAST * idx = nullptr;
    if (mIndexStrm) {
        idx = getInputStreamSet("index")[0];
    }
    for (unsigned i = 0; i < mTransitionCCs.size(); i++) {
        PabloAST * trCC = ccc->compileCC(mTransitionCCs[i], pb);
        PabloAST * transition = pb.createAnd(pb.createIndexedAdvance(trCC, idx, 1), trCC);
        transitions = pb.createOr(transitions, transition);
    }
    PabloAST * mask = pb.createNot(transitions);
    if (mIndexStrm) {
        mask = pb.createAnd(mask, idx);
    }
    pb.createAssign(pb.createExtract(getOutputStreamVar("mask"), pb.getInteger(0)), mask);
}


FindEmptyBreaks::FindEmptyBreaks(LLVMTypeSystemInterface & ts, StreamSet * breaks, StreamSet * empties, StreamSet * index)
: PabloKernel(ts, index == nullptr ? "FindEmptyBreaks" : "FindEmptyBreaks+x",
              {Binding{"breaks", breaks}},
              {Binding{"empties", empties}}), mIndexStrm(nullptr) {
                  if (index != nullptr) {
                      mInputStreamSets.push_back(Binding{"index", index});
                      mIndexStrm = index;
                  }
              }

void FindEmptyBreaks::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * breaks = getInputStreamSet("breaks")[0];
    PabloAST * advBreaks = nullptr;
    if (mIndexStrm) {
        PabloAST * index = getInputStreamSet("index")[0];
        PabloAST * nonBreaks = pb.createAnd(pb.createNot(breaks), index);
        advBreaks = pb.createAnd(pb.createNot(pb.createIndexedAdvance(nonBreaks, index, 1)), index);
    } else {
        advBreaks = pb.createNot(pb.createAdvance(pb.createNot(breaks), 1));
    }
    PabloAST * empties = pb.createAnd(advBreaks, breaks);
    writeOutputStreamSet("empties", std::vector<PabloAST*>{empties});
}
