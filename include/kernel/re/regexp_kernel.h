/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <re/adt/adt.h>
#include <re/alphabet/alphabet.h>
#include <kernel/core/streamset.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <pablo/pablo_kernel.h>
#include <re/transforms/name_intro.h>
#include <map>

using Alphabets = std::vector<std::pair<const cc::Alphabet *, kernel::StreamSet *>>;

enum ExternalStreamKind {ZeroWidth, FixedLength, StartIndexed, EndIndexed};

struct ExternalStream {
    ExternalStreamKind kind;
    unsigned offset;
    std::pair<int, int> lgthRange;
    kernel::StreamSet * extStream;
};

enum class RE_CombiningType {None, Exclude, Include};

using ExternalNameMap = std::map<std::string, ExternalStream>;

class RE_CompilerContext {
    friend class RE_Kernel;
    friend class RE_PipelineBuilder;
public:
    RE_CompilerContext();

    void setCodeUnitContext(const cc::Alphabet * a, kernel::StreamSet * s);

    void setMatchRegions(kernel::StreamSet * starts, kernel::StreamSet * follows);

    void setIndexingContext(const cc::Alphabet * a, kernel::StreamSet * s);

    void addAlphabet(const cc::Alphabet * a, kernel::StreamSet * basis);

    void addExternal(std::string extName, ExternalStream s);
    
    void setCombiningStream(kernel::StreamSet * combiningStream, RE_CombiningType k);

private:
    const cc::Alphabet * mCodeUnitAlphabet;
    kernel::StreamSet * mCodeUnitStream;
    kernel::StreamSet * mMatchStarts;
    kernel::StreamSet * mMatchFollows;
    const cc::Alphabet * mLengthAlphabet;
    kernel::StreamSet * mIndexStream;
    RE_CombiningType mCombiningType;
    kernel::StreamSet * mCombiningStream;
    Alphabets mAlphabets;
    ExternalNameMap mExternals;
    std::vector<std::string> mSpanNames;
};

class RE_Kernel : public pablo::PabloKernel {
public:
    RE_Kernel(LLVMTypeSystemInterface & ts,
              RE_CompilerContext & ctxt, re::RE * re, kernel::StreamSet * results);
    void generatePabloMethod() override;

    std::string makeSignature(RE_CompilerContext & ctxt, re::RE * r);
    kernel::Bindings makeInputBindings(RE_CompilerContext & ctxt, re::RE * r);
    kernel::Bindings makeOutputBindings(re::RE * r, kernel::StreamSet * results);

private:
    RE_CompilerContext mContext;
    re::RE * mRE;
};

class FixedDistanceMatchesKernel : public pablo::PabloKernel {
public:
    FixedDistanceMatchesKernel(LLVMTypeSystemInterface & ts, unsigned distance, 
                               kernel::StreamSet * Basis, kernel::StreamSet * Matches, kernel::StreamSet * ToCheck  = nullptr);
protected:
    void generatePabloMethod() override;
private:
    unsigned mMatchDistance;
    bool mHasCheckStream;
};

class CodePointMatchKernel : public pablo::PabloKernel {
public:
    CodePointMatchKernel(LLVMTypeSystemInterface & ts, 
                         UCD::property_t prop, unsigned distance, 
                         kernel::StreamSet * Basis, kernel::StreamSet * Matches);
protected:
    void generatePabloMethod() override;
private:
    unsigned mMatchDistance;
    UCD::property_t mProperty;
};

class FixedMatchSpansKernel : public pablo::PabloKernel {
public:
    FixedMatchSpansKernel(LLVMTypeSystemInterface & ts, unsigned length, unsigned offset, kernel::StreamSet * MatchMarks, kernel::StreamSet * MatchSpans);
protected:
    void generatePabloMethod() override;
    unsigned mMatchLength;
    unsigned mOffset;
};

class LongestSpan : public pablo::PabloKernel {
public:
    LongestSpan(LLVMTypeSystemInterface & ts, unsigned pfxOffset, unsigned endOffset, 
                kernel::StreamSet * pfxStrm, kernel:: StreamSet * endBack, kernel::StreamSet * matchEnd,
                kernel::StreamSet * spans);
protected:
    void generatePabloMethod() override;
private:
    unsigned mPfxOffset;
    unsigned mEndOffset;
};

class RE_PipelineBuilder {
public:
    RE_PipelineBuilder(kernel::PipelineBuilder & P, RE_CompilerContext & ctxt) :
        mPB(P), mCtxt(ctxt), mCaseInsensitive(false), mMatchSpans(false) {}
    void setCaseInsensitive(bool caseless) {mCaseInsensitive = caseless;}

    void matchSearchPipeline(re::RE * re, kernel::StreamSet * results);
    void matchSpanPipeline(re::RE * re, kernel::StreamSet * matches, kernel::StreamSet * spans);
    kernel::PipelineBuilder & getPipelineBuilder() {return mPB;}

protected:
    // Internal methods.
    void addExternal(std::string extName, ExternalStream s);

    re::RE * prepareRE(re::RE * re);

    re::RE * spanFactoring(re::RE * re);

    re::RE * processReferences(re::RE * re);

    void prepareExternals(re::RE * re);

    void compileExternal(re::Name * name);

    void compileProperty(re::PropertyExpression * pe);

    void getSpan(re::RE * re, kernel::StreamSet * spans);

private:
    kernel::PipelineBuilder & mPB;
    RE_CompilerContext mCtxt;
    bool mCaseInsensitive;
    bool mMatchSpans;
    re::RE * mRE;
    re::UniquePrefixNamer mUPnamer;
};

//
// Create the kernels and pipeline necessary for any Unicode boundary or codepoint property.
// BasisBits may either be 8 basis streams for UTF-8 or 21 streams for Unicode.
// In the case of UTF-8, u8index is required as the IndexStream.
//
void UnicodePropertyLogic(kernel::PipelineBuilder & P, re::PropertyExpression * pe,
                          kernel::StreamSet * BasisBits, kernel::StreamSet * IndexStream,
                          kernel::StreamSet * PropertyStream);
void UnicodePropertyLogic(kernel::PipelineBuilder & P, re::PropertyExpression * pe,
                          kernel::StreamSet * BasisBits, kernel::StreamSet * PropertyStream);
