#include <grep/nested_grep_engine.h>
#include <grep/regex_passes.h>
#include <re/unicode/casing.h>
#include <re/transforms/exclude_CC.h>
#include <re/transforms/to_utf8.h>
#include <re/unicode/re_name_resolve.h>
#include <kernel/io/source_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <re/cc/cc_kernel.h>
#include <kernel/scan/scanmatchgen.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/core/kernel_builder.h>
#include <llvm/Support/raw_ostream.h>
#include <grep/grep_toolchain.h>

#include <re/printer/re_printer.h>

using namespace kernel;
using namespace llvm;

namespace grep {

class CopyBreaksToMatches final : public MultiBlockKernel {
public:

    CopyBreaksToMatches(LLVMTypeSystemInterface & ts,
               StreamSet * const BasisBits,
               StreamSet * const u8index,
               StreamSet * const breaks,
               StreamSet * const matches)
    : MultiBlockKernel(ts
                       , "gitignoreC"
                       // inputs
                       , {{"BasisBits", BasisBits}, {"u8index", u8index}, {"breaks", breaks}}
                       // outputs
                       , {{"matches", matches, FixedRate(), Add1()}}
                       // scalars
                       , {}, {}, {}) {

    }

protected:
    void generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) override {
        PointerType * const int8PtrTy = b.getInt8PtrTy();
        Value * const processed = b.getProcessedItemCount("breaks");
        Value * const source = b.CreatePointerCast(b.getRawInputPointer("breaks", processed), int8PtrTy);
        Value * const produced = b.getProducedItemCount("matches");
        Value * const target = b.CreatePointerCast(b.getRawOutputPointer("matches", produced), int8PtrTy);
        Value * const toCopy = b.CreateMul(numOfStrides, b.getSize(getStride()));
        b.CreateMemCpy(target, source, toCopy, b.getBitBlockWidth() / 8);
    }

};

NestedInternalSearchEngine::NestedInternalSearchEngine(BaseDriver & driver)
: mGrepRecordBreak(GrepRecordBreakKind::LF)
, mCaseInsensitive(false)
, mGrepDriver(driver)
, mNested(1, nullptr) {


}

void NestedInternalSearchEngine::push(const re::PatternVector & patterns) {
    // If we have no patterns and this is the "root" pattern,
    // we'll still need an empty gitignore kernel even if it
    // just returns the record break stream for input.
    // Otherwise just reuse the parent kernel.

    const auto preserve = mGrepDriver.getPreservesKernels();
    mGrepDriver.setPreserveKernels(true);

    auto P = CreatePipeline(mGrepDriver, Input<const char *>{"buffer"}, Input<size_t>{"length"}, Input<MatchAccumulator &>{"accumulator"} );

    Scalar * const buffer = P.getInputScalar("buffer");
    Scalar * const length = P.getInputScalar("length");
    Scalar * const accumulator = P.getInputScalar("accumulator");

    StreamSet * const byteStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<MemorySourceKernel>(buffer, length, byteStream);
    StreamSet * const basisBits = P.CreateStreamSet(8);
    P.CreateKernelCall<S2PKernel>(byteStream, basisBits);
    StreamSet * const breaks = P.CreateStreamSet();

    re::CC * breakCC = nullptr;

    if (mGrepRecordBreak == GrepRecordBreakKind::Null) {
        breakCC = re::makeByte(0x0);
    } else {// if (mGrepRecordBreak == GrepRecordBreakKind::LF)
        breakCC = re::makeByte(0x0A);
    }

    P.CreateKernelCall<CharacterClassKernelBuilder>(std::vector<re::CC *>{breakCC}, basisBits, breaks);
    StreamSet * const U8index = P.CreateStreamSet();
    P.CreateKernelCall<UTF8_index>(basisBits, U8index);

    StreamSet * const matches = P.CreateStreamSet();

    assert (mNested.size() > 0 && mNested[0] == nullptr);
    assert (mNested.size() == 1 || mNested[1] != nullptr);

    Kernel * kernel = nullptr;

    if (LLVM_UNLIKELY(patterns.empty())) {

        if (LLVM_LIKELY(mNested.size() > 1)) {
            kernel = mNested.back(); assert (kernel);
            mNested.push_back(kernel);
        } else {
            kernel = new CopyBreaksToMatches(mGrepDriver,
                                             basisBits, U8index, breaks,
                                             matches);
        }

    } else {

        auto E = CreatePipeline(mGrepDriver,
            Input<streamset_t>{"basis", basisBits},
                                Input<streamset_t>{"u8index", U8index},
                                Input<streamset_t>{"breaks", breaks},
            Output<streamset_t>{"matches", matches, Add1(), ManagedBuffer()},
            InternallySynchronized());

        std::string tmp;
        raw_string_ostream name(tmp);
        name << "gitignore";

        const auto n = patterns.size();
        assert (n > 0);
        SmallVector<Kernel *, 32> pipeline;
        pipeline.reserve(n + 1);

        Kernel * const outerKernel = mNested.back();
        StreamSet * resultSoFar = breaks;
        if (outerKernel) {
            Kernel * const chained = E.AddKernelFamilyCall(outerKernel);
            assert (chained->getNumOfStreamInputs() == 3);
            chained->setInputStreamSetAt(0, basisBits);
            chained->setInputStreamSetAt(1, U8index);
            chained->setInputStreamSetAt(2, breaks);
            pipeline.push_back(chained);
            assert (chained->getNumOfStreamOutputs() > 0);
            resultSoFar = chained->getOutputStreamSet(0); assert (resultSoFar);
        }

        for (unsigned i = 0; i != n; ++i) {
            StreamSet * MatchResults = nullptr;
            if (LLVM_UNLIKELY(i == (n - 1UL))) {
                assert (E.getNumOfStreamOutputs() > 0);
                MatchResults = E.getOutputStreamSet(0);
            } else {
                MatchResults = E.CreateStreamSet();
            }

            auto options = std::make_unique<GrepKernelOptions>();

            auto r = resolveCaseInsensitiveMode(patterns[i].second, mCaseInsensitive);
            r = regular_expression_passes(r);
            r = re::exclude_CC(r, breakCC);
            r = resolveAnchors(r, breakCC);
            r = toUTF8(r);

            options->setRE(r);
            options->setBarrier(breaks);
            options->addAlphabet(&cc::UTF8, basisBits);
            options->setResults(MatchResults);
            // check if we need to combine the current result with the new set of matches
            const bool exclude = (patterns[i].first == re::PatternKind::Exclude);
            if (i || outerKernel || exclude) {
                options->setCombiningStream(exclude ? GrepCombiningType::Exclude : GrepCombiningType::Include, resultSoFar);
            }
            options->addExternal("UTF8_index", U8index);
            Kernel * K = E.CreateKernelFamilyCall<ICGrepKernel>(std::move(options));
            pipeline.push_back(K);
            resultSoFar = MatchResults;

        }
        assert (resultSoFar == E.getOutputStreamSet(0));

        mGrepDriver.generateUncachedKernels();

        for (Kernel * K : pipeline) {
            assert (K->getCompilationStatus() >= Kernel::CompilationStatus::StateConstructed);
            char flags = '0';
            if (LLVM_LIKELY(K->isStateful())) {
                flags |= 1;
            }
            if (LLVM_UNLIKELY(K->hasThreadLocal())) {
                flags |= 2;
            }
            if (LLVM_UNLIKELY(K->allocatesInternalStreamSets())) {
                flags |= 4;
            }
            name << flags;
        }
        name.flush();

        E.setUniqueName(name.str());

        kernel = E.makeKernel();
    }

    P.AddKernelFamilyCall(kernel);

    if (MatchCoordinateBlocks > 0) {
        StreamSet * const MatchCoords = P.CreateStreamSet(3, sizeof(size_t) * 8);
        P.CreateKernelCall<MatchCoordinatesKernel>(matches, breaks, MatchCoords, MatchCoordinateBlocks);
        Kernel * const matchK = P.CreateKernelCall<MatchReporter>(byteStream, MatchCoords, accumulator);
        matchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        matchK->link("finalize_match_wrapper", finalize_match_wrapper);
    } else {
        Kernel * const scanMatchK = P.CreateKernelCall<ScanMatchKernel>(matches, breaks, byteStream, accumulator, ScanMatchBlocks);
        scanMatchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        scanMatchK->link("finalize_match_wrapper", finalize_match_wrapper);
    }

    mNested.push_back(kernel);

    mMainMethod.push_back(P.compile());
    assert (mMainMethod.size() + 1 == mNested.size());

    mGrepDriver.setPreserveKernels(preserve);
}

void NestedInternalSearchEngine::pop() {
    assert (mNested.size() > 1);
    mNested.pop_back();
    assert (mMainMethod.size() > 0);
    mMainMethod.pop_back();
    assert (mMainMethod.size() + 1 == mNested.size());
}

void NestedInternalSearchEngine::doGrep(const char * search_buffer, size_t bufferLength, MatchAccumulator & accum) {
    assert (mMainMethod.size() > 0);
    auto f = mMainMethod.back(); assert (f);
    f(search_buffer, bufferLength, accum);
}

NestedInternalSearchEngine::~NestedInternalSearchEngine() { }


}
