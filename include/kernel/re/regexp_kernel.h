/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <re/adt/adt.h>
#include <re/alphabet/alphabet.h>
#include <kernel/core/streamset.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <pablo/pablo_kernel.h>
#include <map>

using Alphabets = std::vector<std::pair<const cc::Alphabet *, kernel::StreamSet *>>;


enum ExternalStreamKind {ZeroWidth, FixedLength, StartIndexed, EndIndexed};

struct ExternalStream {
    ExternalStreamKind kind;
    unsigned offset;
    kernel::StreamSet * extStream;
};

enum class RE_CombiningType {None, Exclude, Include};

using ExternalNameMap = std::map<std::string, ExternalStream>;

class RE_CompilerContext {
    friend class RE_Kernel;
public:
    RE_CompilerContext();

    void setCodeUnitContext(const cc::Alphabet * a, kernel::StreamSet * s);

    void setBarrier(kernel::StreamSet * s);

    void setIndexingContext(const cc::Alphabet * a, kernel::StreamSet * s);

    void addAlphabet(const cc::Alphabet * a, kernel::StreamSet * basis);

    void addExternal(std::string extName, ExternalStream s);

private:
    const cc::Alphabet * mCodeUnitAlphabet;
    kernel::StreamSet * mCodeUnitStream;
    kernel::StreamSet * mBarrierStream;
    const cc::Alphabet * mIndexingAlphabet;
    kernel::StreamSet * mIndexStream;
    RE_CombiningType mCombiningType;
    kernel::StreamSet * mCombiningStream;
    Alphabets mAlphabets;
    ExternalNameMap mExternals;
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
