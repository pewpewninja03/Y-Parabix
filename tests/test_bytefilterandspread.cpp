/*
 *  Copyright (c) 2018 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include <kernel/core/idisa_target.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/pipeline/pipeline_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/deletion.h>
#include <testing/assert.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <boost/integer/common_factor_rt.hpp>
#include <llvm/IR/Constant.h>
#include <testing/assert.h>
#include <random>
#include <util/aligned_allocator.h>
#include <limits>
#if BOOST_VERSION >= 107600
#include <boost/core/bit.hpp>
#endif



#include <llvm/Support/raw_os_ostream.h>

using namespace kernel;
using namespace llvm;
using namespace testing;
using namespace boost::integer;

enum Mode {
    Filter,
    Spread,
    Any
};

static cl::opt<unsigned> optTestLength("test-length", cl::desc("Source length of each test"), cl::init(0));

static cl::opt<unsigned> optFieldWidth("field-width", cl::desc("Field width of pattern elements"), cl::init(0));

static cl::opt<Mode>
optMode("mode", cl::init(Mode::Any), cl::desc("Set the front-end optimization level:"),
                  cl::values(clEnumValN(Mode::Filter, "filter", "no optimizations (default)"),
                             clEnumValN(Mode::Spread, "spread", "trivial optimizations"),
                             clEnumValN(Mode::Any, "any", "standard optimizations")));

static cl::opt<unsigned> optStreamCount("stream-count", cl::desc("Number of streams per streamset"), cl::init(0));

static cl::opt<unsigned> optTrials("trials", cl::desc("Number of tests to execute"), cl::init(10));


static cl::opt<bool> optVerbose("v", cl::desc("Print verbose output"), cl::init(false));


#if BOOST_VERSION < 107600
template <typename T> int scan_forward_zeroes(const T x) noexcept;
template <> inline int scan_forward_zeroes<unsigned int>(const unsigned int x) noexcept { return __builtin_ctz(x); }
template <> inline int scan_forward_zeroes<unsigned long>(const unsigned long x) noexcept { return __builtin_ctzl(x); }
template <> inline int scan_forward_zeroes<unsigned long long>(const unsigned long long x) noexcept { return __builtin_ctzll(x); }

template <typename T> unsigned get_popcount(const T x) noexcept;
template <> inline unsigned get_popcount<unsigned int>(const unsigned int x) noexcept { return __builtin_popcount(x); }
template <> inline unsigned get_popcount<unsigned long>(const unsigned long x) noexcept { return __builtin_popcountl(x); }
template <> inline unsigned get_popcount<unsigned long long>(const unsigned long long x) noexcept { return __builtin_popcountll(x); }

#else
template <typename T> int scan_forward_zeroes(const T x) noexcept {
    return boost::core::countr_zero<T>(x);
}
template <typename T> unsigned get_popcount(const T x) noexcept {
    return boost::core::popcount<T>(x);
}
#endif




template<size_t FieldWidth>
uint32_t runTestCase(CPUDriver & driver, const size_t streamCount, const size_t testLength, const Mode mode, std::default_random_engine & rng) {

    using datatype_t = typename boost::uint_t<FieldWidth>::exact;

    using Allocator = AlignedAllocator<uint8_t, (512 / 8)>;

    if (LLVM_UNLIKELY(streamCount == 0 || testLength == 0)) {
        return 0U;
    }

    const auto blockWidth = driver.getBitBlockWidth();

    assert ((blockWidth % 64) == 0 && blockWidth > 64);

    Allocator alloc;

    size_t sourceDataLength = (testLength + 511ULL) & ~511ULL;

    datatype_t * const source = (datatype_t*)alloc.allocate(sourceDataLength * streamCount * sizeof(datatype_t));

    std::uniform_int_distribution<datatype_t> dataDist(0ULL, std::numeric_limits<datatype_t>::max());

    assert ((sourceDataLength % blockWidth) == 0);

    for (size_t i = 0, out = 0; i < sourceDataLength; i += blockWidth) {

        if (LLVM_LIKELY((i + blockWidth) <= testLength)) {
            for (size_t j = 0; j < (streamCount * blockWidth); ++j) {
                source[out++] = dataDist(rng);
            }
        } else if (i < testLength) {
            const auto m = testLength - i;
            for (size_t j = 0; j < streamCount; ++j) {
                for (size_t k = 0; k < m; ++k) {
                    source[out++] = dataDist(rng);
                }
                for (size_t k = m; k < blockWidth; ++k) {
                    source[out++] = 0;
                }
            }
        } else {
            for (size_t j = 0; j < (streamCount * blockWidth); ++j) {
                source[out++] = 0;
            }
        }
    }

    std::uniform_int_distribution<uint64_t> markDist(0ULL, std::numeric_limits<uint64_t>::max());

    uint64_t * markers = nullptr;

    size_t markerLength = 0;

    datatype_t * result = nullptr;

    size_t resultLength = 0;


    if (mode == Mode::Filter) {

        markerLength = (testLength + 63) / 64;

        markers = (uint64_t*)alloc.allocate(markerLength * sizeof(uint64_t));

        size_t popCount = 0;

        for (size_t i = 0; i < markerLength; ++i) {
            uint64_t v = markDist(rng);
            while (LLVM_UNLIKELY(popCount > testLength)) {
                // remove rightmost bit
                assert (v);
                v &= (v - 1);
                --popCount;
            }
            assert (popCount <= testLength);
            markers[i] = v;
            popCount += get_popcount(v);
        }

        resultLength = (popCount + 511ULL) & ~511ULL;

        const auto resultBytes = resultLength * streamCount * sizeof(datatype_t);

        result = (datatype_t*)alloc.allocate(resultBytes);
        std::memset(result, 0, resultBytes);

        const auto markersPerBlock = (blockWidth / 64);

        for (size_t j = 0; j < streamCount; ++j) {
            size_t out = j * blockWidth;
            for (size_t i = 0; i < markerLength; ++i) {
                auto v = markers[i];
                const auto base = (j * blockWidth) + ((i % markersPerBlock) * 64) + ((i / markersPerBlock) * streamCount * blockWidth);
                while (v) {
                    const auto p = scan_forward_zeroes(v);
                    assert ((v & (1ULL << p)) != 0);
                    v ^= (1ULL << p);
                    assert (result[out] == 0);
                    assert ((base | p) == (base + p));
                    result[out++] = source[base | p];
                    if ((out % blockWidth) == 0) {
                        out += (streamCount - 1) * blockWidth;
                    }
                }
            }
        }

        markerLength = testLength;

    } else if (mode == Mode::Spread) {

        size_t popCount = 0;

        size_t capacity = (testLength * 4  + 511) / 64;

        markers = (uint64_t*)alloc.allocate(capacity * sizeof(uint64_t));

        size_t total = 0;

        while (popCount < testLength) {

            uint64_t v = markDist(rng);
            popCount += get_popcount(v);

            if (LLVM_UNLIKELY(total >= capacity)) {
                // resize
                const auto newCapacity = (capacity * 2);
                uint64_t * const newMarkers = (uint64_t*)alloc.allocate(newCapacity * sizeof(uint64_t));
                std::memcpy(newMarkers, markers, capacity * sizeof(uint64_t));
                markers = newMarkers;
                capacity = newCapacity;
            }

            while (LLVM_UNLIKELY(popCount > testLength)) {
                // remove rightmost bit
                assert (v);
                v &= (v - 1);
                --popCount;
            }
            markers[total++] = v;
        }

        assert (popCount == testLength);

        markerLength = (total * 64);
        resultLength = (markerLength + 511) & ~511;
        const auto resultBytes = resultLength * streamCount * sizeof(datatype_t);
        result = (datatype_t*)alloc.allocate(resultBytes);
        std::memset(result, 0, resultBytes);

        const auto markersPerBlock = (blockWidth / 64);

        for (size_t j = 0; j < streamCount; ++j) {
            size_t in = j * blockWidth;
            for (size_t i = 0; i < total; ++i) {
                uint64_t v = markers[i];
                const auto base = (j * blockWidth) + ((i % markersPerBlock) * 64) + ((i / markersPerBlock) * streamCount * blockWidth);
                while (v) {
                    const auto p = scan_forward_zeroes(v);
                    assert ((v & (1ULL << p)) != 0);
                    v ^= (1ULL << p);
                    const size_t nextOut = base | p;
                    assert (result[nextOut] == 0);
                    result[nextOut] = source[in++];
                    if ((in % blockWidth) == 0) {
                        in += (streamCount - 1) * blockWidth;
                    }
                }
            }
        }

    }

    auto P = CreatePipeline(driver,
                            Input<streamset_t>{"source", streamCount, FieldWidth}, Input<streamset_t>{"markers", 1, 1},
                            Input<streamset_t>{"result", streamCount, FieldWidth},
                            Input<uint32_t &>{"output"});

    StreamSet * generated = P.CreateStreamSet(streamCount, FieldWidth);

    if (mode == Mode::Filter) {
        P.CreateKernelCall<ByteFilterByMaskKernel>(P.getInputStreamSet(0), P.getInputStreamSet(1), generated);
    } else if (mode == Mode::Spread) {
        P.CreateKernelCall<ByteSpreadByMaskKernel>(P.getInputStreamSet(0), P.getInputStreamSet(1), generated);
    }

    Scalar * invalid = P.getInputScalar(0); assert (invalid);

    StreamSet * toMatch = P.getInputStreamSet(2); assert (toMatch);

    P.CreateKernelCall<StreamEquivalenceKernel>(StreamEquivalenceKernel::Mode::EQ, generated, toMatch, invalid);

    auto func = P.compile();

    StreamSetPtr pSource{source, testLength};
    StreamSetPtr pMarkers{markers, markerLength};
    StreamSetPtr pResult{result, resultLength};

    uint32_t retVal = 0;

    func(pSource, pMarkers, pResult, retVal);

    return retVal;
}


bool runTestCase(CPUDriver & driver, const size_t streamCount, const size_t testLength, const size_t fieldWidth, const Mode mode, std::default_random_engine & rng) {

    uint32_t result = 0;

    try {
        if (fieldWidth == 8) {
            result = runTestCase<8>(driver, streamCount, testLength, mode, rng);
        } else if (fieldWidth == 16) {
            result = runTestCase<16>(driver, streamCount, testLength, mode, rng);
        } else if (fieldWidth == 32) {
            result = runTestCase<32>(driver, streamCount, testLength, mode, rng);
        } else if (fieldWidth == 64) {
            result = runTestCase<64>(driver, streamCount, testLength, mode, rng);
        } else {
            llvm::report_fatal_error("Unexpected field width");
        }
    }  catch (...) {
        result = 1;
    }

    if (result != 0 || optVerbose) {
        if (mode == Mode::Filter) {
            llvm::errs() << "FILTER";
        } else if (mode == Mode::Spread) {
            llvm::errs() << "SPREAD";
        }

        llvm::errs() << " TEST: " << streamCount << 'x' << testLength << 'x' << fieldWidth << " -- ";
        if (result == 0) {
            llvm::errs() << "success\n";
        } else {
            llvm::errs() << "failed\n";
            exit(-1);
        }
    }

    return (result != 0);
}

bool runTestCase(CPUDriver & driver, std::default_random_engine & rng) {

    size_t streamCount = 0;
    if (optStreamCount.getNumOccurrences()) {
        streamCount = optStreamCount.getValue();
    } else {
        std::uniform_int_distribution<size_t> scDist(1, 3);
        streamCount = scDist(rng);
    }

    size_t testLength = 0;
    if (optTestLength.getNumOccurrences()) {
        testLength = optTestLength.getValue();
    } else {
        std::uniform_int_distribution<size_t> tlDist(700LL, 3000ULL);
        testLength = tlDist(rng);
    }

    size_t fieldWidth = 0;
    if (optFieldWidth.getNumOccurrences()) {
        fieldWidth = optFieldWidth.getValue();
    } else {
        std::uniform_int_distribution<size_t> fwDist(3ULL, 6ULL);
        fieldWidth = 1ULL << (fwDist(rng));
    }

    Mode mode;
    if (optMode.getValue() == Mode::Any) {
        std::uniform_int_distribution<unsigned> modeDist(Mode::Filter, Mode::Spread);
        mode = (Mode)modeDist(rng);
    } else {
        mode = optMode.getValue();
    }

    return runTestCase(driver, streamCount, testLength, fieldWidth, mode, rng);
}


int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {});
    CPUDriver driver("test");
    std::random_device rd;
    std::default_random_engine rng(rd());
    for (unsigned rounds = 0; rounds < optTrials; ++rounds) {
        if (runTestCase(driver, rng)) {
            return -1;
        }
    }
    return 0;
}
