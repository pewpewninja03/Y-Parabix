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

class ExternalStreamObject;

//  StreamIndexCode is used to identify the indexing base of
//  streamsets.   UTF-8 and Unicode indexed streams are two
//  common indexing bases.
using StreamIndexCode = unsigned;

struct StreamIndexInfo {
    std::string name;
    StreamIndexCode base;
    std::string indexStreamName;
};

class ExternalStreamTable {
public:
    ExternalStreamTable() = default;
    StreamIndexCode declareStreamIndex(std::string indexName, StreamIndexCode base = 0, std::string indexStreamName = "");
    StreamIndexCode getStreamIndex(std::string indexName);
    void declareExternal(StreamIndexCode c, std::string externalName, ExternalStreamObject * ext);
    ExternalStreamObject * lookup(StreamIndexCode c, std::string externalName);
    bool isDeclared(StreamIndexCode c, std::string externalName);
    bool hasReferenceTo(StreamIndexCode c, std::string externalName);
    StreamSet * getStreamSet(PipelineBuilder & b, StreamIndexCode c, std::string externalName);
    void resetExternals();  // Reset all externals to unresolved.
    void resolveExternals(PipelineBuilder &b);
    ~ExternalStreamTable();
private:
    std::vector<StreamIndexInfo> mStreamIndices;
    std::vector<std::map<std::string, ExternalStreamObject *>> mExternalMap; // <-- memory leak
};

using ExternalMapRef = ExternalStreamTable *;

class ExternalStreamObject {
    friend class ExternalStreamTable;
public:
    enum class Kind : unsigned {
        U21, PreDefined, LineStarts, CC_External, RE_External,
        PropertyExternal, PropertyBasis, PropertyDistance, PropertyBoundary,
        WordBoundaryExternal, GraphemeClusterBreak, Multiplexed,
        FilterByMask, FixedSpan, MarkedSpanExternal,
        CCmask, CCselfTransitionMask, MaskedFixedSpan
    };
    inline Kind getKind() const {
        return mKind;
    }
    StreamSet * getStreamSet() const {
        return mStreamSet;
    }
    // Most externals are computed from the basis bit stremas.
    virtual const std::vector<std::string> getParameters() {
        return std::vector<std::string>{"basis"};
    }
    virtual void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) = 0;
    std::pair<int, int> getLengthRange() const {return mLengthRange;}
    int getOffset() const {return mOffset;}
    bool isResolved() const {return mStreamSet != nullptr;}
    virtual ~ExternalStreamObject() {}
protected:
    ExternalStreamObject(Kind k, std::pair<int, int> lgthRange = std::make_pair(1,1), int offset = 0) :
        mKind(k), mLengthRange(lgthRange), mOffset(offset), mStreamSet(nullptr)  {}
    void installStreamSet(StreamSet * s);
protected:
    const Kind mKind;
    const std::pair<int, int> mLengthRange;
    const int mOffset;
    StreamSet * mStreamSet;
};

class PreDefined final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::PreDefined;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override {return {};}
    PreDefined(StreamSet * predefined, std::pair<int, int> lgthRange = std::make_pair(1,1), int offset = 0) :
        ExternalStreamObject(Kind::PreDefined, lgthRange, offset) {mStreamSet = predefined;}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override {}
};

class LineStartsExternal final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::LineStarts;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override;
    LineStartsExternal(std::vector<std::string> parms = {"$"}) :
        ExternalStreamObject(Kind::LineStarts, std::make_pair(0, 0), 1), mParms(parms) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    const std::vector<std::string> mParms;
};

class U21_External final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::U21;
    }
    static inline bool classof(const void *) {
        return false;
    }
    U21_External() : ExternalStreamObject(Kind::U21) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
};

class PropertyExternal final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::PropertyExternal;
    }
    static inline bool classof(const void *) {
        return false;
    }
    PropertyExternal(re::Name * n) :
        ExternalStreamObject(Kind::PropertyExternal), mName(n) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    re::Name * const mName;
};

class PropertyBoundaryExternal final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::PropertyBoundary;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override;
    PropertyBoundaryExternal(UCD::property_t p) :
        ExternalStreamObject(Kind::PropertyBoundary, std::make_pair(0, 0), 1), mProperty(p) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    const UCD::property_t mProperty;
};

class CC_External final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::CC_External;
    }
    static inline bool classof(const void *) {
        return false;
    }
    CC_External(re::CC * cc) :
        ExternalStreamObject(Kind::CC_External), mCharClass(cc) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    re::CC * mCharClass;
};

class RE_External final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::RE_External;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override {
        return mParams;
    }
    RE_External(grep::GrepEngine * engine, re::RE * re, const cc::Alphabet * a) :
        ExternalStreamObject(Kind::RE_External, re::getLengthRange(re, a), grepOffset(re)),
            mGrepEngine(engine), mRE(re), mIndexAlphabet(a), mParams(re::gatherExternals(re)) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    grep::GrepEngine * const mGrepEngine;
    re::RE * const mRE;
    const cc::Alphabet * const mIndexAlphabet;
    const std::vector<std::string> mParams;
};

class PropertyDistanceExternal final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::PropertyDistance;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override;
    PropertyDistanceExternal(UCD::property_t p, unsigned dist) :
        ExternalStreamObject(Kind::PropertyDistance),
        mProperty(p), mDistance(dist) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    const UCD::property_t mProperty;
    const unsigned mDistance;
};

class GraphemeClusterBreak final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::GraphemeClusterBreak;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override;
    GraphemeClusterBreak(grep::GrepEngine * engine, const cc::Alphabet * a) :
        ExternalStreamObject(Kind::GraphemeClusterBreak, std::make_pair(0, 0), 1), mGrepEngine(engine), mIndexAlphabet(a)  {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    grep::GrepEngine * const  mGrepEngine;
    const cc::Alphabet * const mIndexAlphabet;
};

class WordBoundaryExternal final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::WordBoundaryExternal;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override;
    WordBoundaryExternal() :
        ExternalStreamObject(Kind::WordBoundaryExternal, std::make_pair(0, 0), 1) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
};

class PropertyBasisExternal final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::PropertyBasis;
    }
    static inline bool classof(const void *) {
        return false;
    }
    PropertyBasisExternal(UCD::property_t p) :
    ExternalStreamObject(Kind::PropertyBasis), mProperty(p) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
     const UCD::property_t mProperty;
};

class MultiplexedExternal final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::Multiplexed;
    }
    static inline bool classof(const void *) {
        return false;
    }
    MultiplexedExternal(cc::MultiplexedAlphabet * mpx) :
        ExternalStreamObject(Kind::Multiplexed), mAlphabet(mpx) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    cc::MultiplexedAlphabet * const mAlphabet;
};

class FilterByMaskExternal final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::FilterByMask;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override {
        return mParamNames;
    }
    StreamIndexCode getBaseIndex() {
        return mBase;
    }
    FilterByMaskExternal(StreamIndexCode base, std::vector<std::string> paramNames, ExternalStreamObject * e) :
        ExternalStreamObject(Kind::FilterByMask, e->getLengthRange(), e->getOffset()),
            mBase(base), mParamNames(paramNames) {} // , mBaseExternal(e)
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    const StreamIndexCode mBase;
    const std::vector<std::string> mParamNames;
//    ExternalStreamObject * const mBaseExternal;
};

class FixedSpanExternal final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::FixedSpan;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override;
    FixedSpanExternal(std::string matchMarks, unsigned lgth, int offset) :
        ExternalStreamObject(Kind::FixedSpan, std::make_pair(lgth, lgth), offset), mMatchMarks(matchMarks) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    const std::string mMatchMarks;
};

class MarkedSpanExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::MarkedSpanExternal;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override;
    MarkedSpanExternal(std::string prefixMarks, unsigned prefixLgth, std::string matchEnds, unsigned offset) :
        ExternalStreamObject(Kind::MarkedSpanExternal, std::make_pair(prefixLgth, INT_MAX), offset),
        mPrefixMarks(prefixMarks), mPrefixLength(prefixLgth), mMatchMarks(matchEnds) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    const std::string mPrefixMarks;
    const unsigned mPrefixLength;
    const std::string mMatchMarks;
};

class CCmask final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::CCmask;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override;
    CCmask(const cc::Alphabet * indexAlphabet, re::CC * CC_to_mask) :
        ExternalStreamObject(Kind::CCmask, std::make_pair(1, 1)),
            mIndexAlphabet(indexAlphabet), mCC_to_mask(CC_to_mask) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    const cc::Alphabet * mIndexAlphabet;
    re::CC * mCC_to_mask;
};

class CCselfTransitionMask final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::CCselfTransitionMask;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override;
    CCselfTransitionMask(const std::vector<re::CC *> transitionCCs) :
        ExternalStreamObject(Kind::CCselfTransitionMask, std::make_pair(1, 1)),  mTransitionCCs(transitionCCs) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    const std::vector<re::CC *> mTransitionCCs;
};

//
// The MaskedFixedSpanExternal is used to determine the spans of regular
// expression matches, when there are an identifiable fixed set of
// positions that must be matched during the matching process, including
// a fixed position that marks the match start.
//
// The parameters of the external are the name of the mask stream identifying
// the fixed positions, the name of the matches stream that identifies
// RE matches, the fixed lookahead distance between the first fixed position
// of the match and the final match position, and the offset distance between
// the first fixed position and the actual match start position.
//
class MaskedFixedSpanExternal final : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::MaskedFixedSpan;
    }
    static inline bool classof(const void *) {
        return false;
    }
    const std::vector<std::string> getParameters() override;
    MaskedFixedSpanExternal(std::string mask, std::string matches, unsigned lgth, int offset) :
        ExternalStreamObject(Kind::MaskedFixedSpan, std::make_pair(lgth, lgth), offset), mMask(mask), mMatches(matches) {}
    void resolveStreamSet(PipelineBuilder & b, std::vector<StreamSet *> inputs) override;
private:
    const std::string mMask;
    const std::string mMatches;
};

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

class FixedMatchSpansKernel : public pablo::PabloKernel {
public:
    FixedMatchSpansKernel(LLVMTypeSystemInterface & ts, unsigned length, unsigned offset, StreamSet * MatchMarks, StreamSet * MatchSpans);
protected:
    void generatePabloMethod() override;
    unsigned mMatchLength;
    unsigned mOffset;
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

class FixedDistanceMatchesKernel : public pablo::PabloKernel {
public:
    FixedDistanceMatchesKernel(LLVMTypeSystemInterface & ts, unsigned distance, StreamSet * Basis, StreamSet * Matches, StreamSet * ToCheck  = nullptr);
protected:
    void generatePabloMethod() override;
private:
    unsigned mMatchDistance;
    bool mHasCheckStream;
};

class CodePointMatchKernel : public pablo::PabloKernel {
public:
    CodePointMatchKernel(LLVMTypeSystemInterface & ts, UCD::property_t prop, unsigned distance, StreamSet * Basis, StreamSet * Matches);
protected:
    void generatePabloMethod() override;
private:
    unsigned mMatchDistance;
    UCD::property_t mProperty;
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

void WordBoundaryLogic(PipelineBuilder & P,
                          StreamSet * Source, StreamSet * U8index, StreamSet * wordBoundary_stream);

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
