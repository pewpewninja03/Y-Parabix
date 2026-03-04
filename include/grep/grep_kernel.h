/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <pablo/pablo_toolchain.h>
#include <re/alphabet/alphabet.h>
#include <re/alphabet/multiplex_CCs.h>
#include <re/analysis/capture-ref.h>
#include <re/analysis/re_analysis.h>
#include <re/analysis/re_name_gather.h>
#include <re/transforms/to_utf8.h>
#include <kernel/pipeline/program_builder.h>
#include <util/slab_allocator.h>

namespace IDISA { class IDISA_Builder; }
namespace cc { class Alphabet; }
namespace re { class CC; class RE; }
namespace grep { class GrepEngine; }

namespace kernel {


enum class GrepCombiningType {None, Exclude, Include};
class GrepKernelOptions {
    friend class ICGrepKernel;
public:
    using Alphabets = std::vector<std::pair<const cc::Alphabet *, StreamSet *>>;
    GrepKernelOptions(const cc::Alphabet * codeUnitAlphabet = &cc::UTF8);
    void setBarrier(StreamSet * barrierStream);
    void setIndexing(StreamSet * indexStream);
    void setCombiningStream(GrepCombiningType t, StreamSet * toCombine);
    void setResults(StreamSet * r);
    void addExternal(std::string name,
                     StreamSet * strm,
                     unsigned offset = 0,
                     std::pair<int, int> lengthRange = std::make_pair<int,int>(1, 1));
    void addAlphabet(const cc::Alphabet * a, StreamSet * basis);
    void setRE(re::RE * re);

protected:
    Bindings makeStreamSetInputBindings();
    Bindings makeStreamSetOutputBindings();
    std::string makeSignature();

private:

    const cc::Alphabet *        mCodeUnitAlphabet = nullptr;
    StreamSet *                 mBarrierStream = nullptr;
    StreamSet *                 mIndexStream = nullptr;
    GrepCombiningType           mCombiningType = GrepCombiningType::None;
    StreamSet *                 mCombiningStream = nullptr;
    StreamSet *                 mResults = nullptr;
    Bindings                    mExternalBindings;
    std::vector<unsigned>       mExternalOffsets;
    std::vector<std::pair<int, int>>       mExternalLengths;
    Alphabets                   mAlphabets;
    re::RE *                    mRE = nullptr;
};


class ICGrepKernel : public pablo::PabloKernel {
public:
    ICGrepKernel(LLVMTypeSystemInterface & ts,
                 std::unique_ptr<GrepKernelOptions> && options);
    llvm::StringRef getSignature() const override;
    bool hasSignature() const override { return true; }
    unsigned getOffset() {return mOffset;}
protected:
    void generatePabloMethod() override;
private:
    ICGrepKernel(LLVMTypeSystemInterface & ts,
                 std::string && optionsSignature,
                 Bindings && inputStreamSetBindings,
                 Bindings && outputStreamSetBindings,
                 std::unique_ptr<GrepKernelOptions> && options);
private:
    const std::unique_ptr<GrepKernelOptions>  mOptions;
    const std::string                         mSignature;
    const unsigned                            mOffset;
};

class MatchedLinesKernel : public pablo::PabloKernel {
public:
    MatchedLinesKernel(LLVMTypeSystemInterface & ts, StreamSet * OriginalMatches, StreamSet * LineBreakStream, StreamSet * Matches);
protected:
    void generatePabloMethod() override;
};

class InvertMatchesKernel : public BlockOrientedKernel {
public:
    InvertMatchesKernel(LLVMTypeSystemInterface & ts, StreamSet * OriginalMatches, StreamSet * LineBreakStream, StreamSet * Matches);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
};

//
//  Given an input stream consisting of spans of 1s, return a pair of
//  streams marking the starts of each span as well as the follows.
//
//  Ex:  spans   .....1111.......1111111.........1....
//       starts  .....1..........1...............1....
//       follows .........1.............1.........1...
//
class SpansToMarksKernel : public pablo::PabloKernel {
public:
    SpansToMarksKernel(LLVMTypeSystemInterface & ts, StreamSet * Spans, StreamSet * EndMarks);
protected:
    void generatePabloMethod() override;
};

class PopcountKernel : public pablo::PabloKernel {
public:
    PopcountKernel(LLVMTypeSystemInterface & ts, StreamSet * const toCount, Scalar * countResult);
protected:
    void generatePabloMethod() override;
};

class AbortOnNull final : public MultiBlockKernel {
public:
    AbortOnNull(LLVMTypeSystemInterface & ts, StreamSet * const InputStream, StreamSet * const OutputStream, Scalar * callbackObject);
private:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) final;

};

/* Given a marker position P, a before-context B and and after-context A, a
   context span is a set of consecutive 1 bits from positions P-B to P+A.

   This kernel computes a coalesced context span stream for all markers in
   a given marker stream.   Coalesced spans occur when markers are separated
   by A + B positions or fewer. */

class ContextSpan final : public pablo::PabloKernel {
public:
    ContextSpan(LLVMTypeSystemInterface & ts, StreamSet * const markerStream, StreamSet * const contextStream, unsigned before, unsigned after);
protected:
    void generatePabloMethod() override;
private:
    const unsigned          mBeforeContext;
    const unsigned          mAfterContext;
};

void GraphemeClusterLogic(PipelineBuilder & P,
                          StreamSet * Source, StreamSet * U8index, StreamSet * GCBstream);

void SimpleWordBoundaryLogic(PipelineBuilder & P,
                          StreamSet * Source, StreamSet * U8index, StreamSet * wordBoundary_stream);

void Level2WordBoundaryLogic(PipelineBuilder & P,
                          StreamSet * Source, StreamSet * wordBoundary_stream);

//  The LongestMatchMarks kernel computes longest-match spans in start-end space.
//  Logically, the input is a set of 2 streams marking, respectively, matches
//  of a necessary prefix of the RE, and matches to the full RE.   However,
//  a single combined stream may be provided as the start-end stream when these
//  two cases do not intersect.   In this case,each 0 bit in start-end space
//  marks the occurrence of a necessary prefix of the RE, while each 1 bit marks
//  an actual match end for the full RE.  The results produced are (a) the start
//  position immediately preceding a full match, and (b) the longest full match
//  corresponding to that start position.
//  For example:
//  start-end stream:  00111110001100101000100
//  (a) starts:        .1.......1...1.1...1...
//  (b) longest end:   ......1....1..1.1...1..
//  The output is a set of two streams for the start and longest end marks, respectively.

class LongestMatchMarks final : public pablo::PabloKernel {
public:
    LongestMatchMarks(LLVMTypeSystemInterface & ts, StreamSet * start_ends, StreamSet * marks);
protected:
    void generatePabloMethod() override;
};

//  Compute match spans given a set of two streams marking a fixed prefix position
//  of the match, as well as the final position of the match.   The prefix may
//  be at an offset from the actual start position of the match, while the suffix
//  may be at an offset from the last matched position.
//  For example, the pair of input streams:
//  prefix:  ...1......1.......1.....
//  final:   .....1.....1......1.....
//  the spans computed with a prefix offset of 2 and suffix offset of 0 are:
//  spans    .11111..1111....111.....
//
class InclusiveSpans final : public pablo::PabloKernel {
public:
    InclusiveSpans(LLVMTypeSystemInterface & ts, unsigned prefixOffset, unsigned suffixOffset,
                   StreamSet * marks, StreamSet * spans);
protected:
    void generatePabloMethod() override;
private:
    unsigned mPrefixOffset;
    unsigned mSuffixOffset;
};

class MaskCC final : public pablo::PabloKernel {
public:
    MaskCC(LLVMTypeSystemInterface & ts, re::CC * CC_to_mask, StreamSet * basis, StreamSet * mask, StreamSet * index = nullptr);
protected:
    void generatePabloMethod() override;
private:
    re::CC * mCC_to_mask;
    StreamSet * mIndexStrm;
};

//  Compute a mask stream for filtering out self transitions of
//  one or more character classes.   A self transition is a set
//  of consecutive occurrences of members of a given character class.
//  The index stream, if present, marks positions considered as
//  full character positions, otherwise the all positions are
//  considered full character positions (equivalent to an index
//  stream of all ones).
//
//  The computed mask stream will consist of a 0 bit at each
//  character position such that it marks a character in the
//  given class and the immediate prior character posiiton
//  also marks a character in the same class.

class MaskSelfTransitions final : public pablo::PabloKernel {
public:
    MaskSelfTransitions(LLVMTypeSystemInterface & ts, const std::vector<re::CC *> transitionCCs,
                        StreamSet * basis, StreamSet * mask, StreamSet * index = nullptr);
protected:
    void generatePabloMethod() override;
private:
    const std::vector<re::CC *> mTransitionCCs;
    StreamSet * mIndexStrm;
};
}
