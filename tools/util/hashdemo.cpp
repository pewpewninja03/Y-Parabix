/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/streamutils/deletion.h>                      // for DeletionKernel
#include <kernel/io/source_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/basis/s2p_kernel.h>                    // for S2PKernel
#include <kernel/io/stdout_kernel.h>                 // for StdOutKernel_
#include <kernel/streamutils/pdep_kernel.h>
#include <llvm/IR/Function.h>                      // for Function, Function...
#include <llvm/IR/Module.h>                        // for Module
#include <llvm/Support/CommandLine.h>              // for ParseCommandLineOp...
#include <llvm/Support/Debug.h>                    // for dbgs
#include <pablo/pablo_kernel.h>                    // for PabloKernel
#include <pablo/parse/pablo_source_kernel.h>
#include <pablo/parse/pablo_parser.h>
#include <pablo/parse/simple_lexer.h>
#include <pablo/parse/rd_parser.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <grep/grep_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/unicode/resolve_properties.h>
#include <kernel/core/kernel_builder.h>
#include <pablo/pe_zeroes.h>
#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/core/streamset.h>
#include <kernel/scan/index_generator.h>
#include <kernel/scan/reader.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/streamutils/streams_merge.h>
#include <kernel/util/bixhash.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pe_zeroes.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <unicode/utf/utf_compiler.h>
#include <re/unicode/resolve_properties.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <fcntl.h>
#include <iostream>
#include <iomanip>
#include <kernel/pipeline/program_builder.h>

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

using namespace pablo;
using namespace parse;
using namespace kernel;
using namespace llvm;
using namespace codegen;

static cl::OptionCategory HashDemoOptions("Hash Demo Options", "Hash demo options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(HashDemoOptions));


class WordMarkKernel : public pablo::PabloKernel {
public:
    WordMarkKernel(LLVMTypeSystemInterface & ts, StreamSet * BasisBits, StreamSet * WordMarks);
protected:
    void generatePabloMethod() override;
};

WordMarkKernel::WordMarkKernel(LLVMTypeSystemInterface & ts, StreamSet * BasisBits, StreamSet * WordMarks)
: PabloKernel(ts, "WordMarks" + UTF::kernelAnnotation(), {Binding{"source", BasisBits}}, {Binding{"WordMarks", WordMarks}}) { }

void WordMarkKernel::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    re::RE * word_prop = re::makePropertyExpression("word");
    word_prop = UCD::linkAndResolve(word_prop);
    re::CC * word_CC = cast<re::CC>(cast<re::PropertyExpression>(word_prop)->getResolvedRE());
    Var * wordChar = pb.createVar("word");
    UTF::UTF_Compiler unicodeCompiler(getInputStreamVar("source"), pb);
    unicodeCompiler.compile({wordChar}, {word_CC});
    pb.createAssign(pb.createExtract(getOutputStreamVar("WordMarks"), pb.getInteger(0)), wordChar);
}

class ParseSymbols : public pablo::PabloKernel {
public:
    ParseSymbols(LLVMTypeSystemInterface & ts,
                StreamSet * basisBits, StreamSet * wordChar, StreamSet * symbolRuns)
    : pablo::PabloKernel(ts, "ParseSymbols",
                         {Binding{"basisBits", basisBits, FixedRate(1), LookAhead(1)},
                             Binding{"wordChar", wordChar, FixedRate(1), LookAhead(3)}},
                         {Binding{"symbolRuns", symbolRuns}}) { }
protected:
    void generatePabloMethod() override;
};

void ParseSymbols::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basisBits");
    cc::Parabix_CC_Compiler_Builder ccc(basis);
    pablo::PabloAST * wordChar = getInputStreamSet("wordChar")[0];
    // Find start bytes of word characters.
    PabloAST * ASCII = ccc.compileCC(re::makeCC(0x0, 0x7F), pb);
    PabloAST * prefix2 = ccc.compileCC(re::makeCC(0xC2, 0xDF), pb);
    PabloAST * prefix3 = ccc.compileCC(re::makeCC(0xE0, 0xEF), pb);
    PabloAST * prefix4 = ccc.compileCC(re::makeCC(0xF0, 0xF4), pb);
    PabloAST * wc1 = pb.createAnd(ASCII, wordChar);
    // Prefixes of word characters
    PabloAST * wprefix2 = pb.createAnd(prefix2, pb.createLookahead(wordChar, 1));
    PabloAST * wprefix3 = pb.createAnd(prefix3, pb.createLookahead(wordChar, 2));
    PabloAST * wprefix4 = pb.createAnd(prefix4, pb.createLookahead(wordChar, 3));
    wc1 = pb.createOr3(wc1, wprefix2, pb.createOr(wprefix3, wprefix4));
    //
    PabloAST * wordStart = pb.createAnd(pb.createNot(pb.createAdvance(wordChar, 1)), wc1, "wordStart");
    PabloAST * wordByte = pb.createOr(wc1, wordChar);
    // Interior bytes of 3 and 4 byte sequences.
    wordByte = pb.createOr(wordByte, pb.createAdvance(pb.createOr(wprefix3, wprefix4), 1));
    wordByte = pb.createOr(wordByte, pb.createAdvance(wprefix4, 2));
    // runs are the bytes after a start symbol until the next symStart byte.
    pablo::PabloAST * runs = pb.createInFile(pb.createAnd(pb.createNot(wordStart), wordByte));
    pb.createAssign(pb.createExtract(getOutputStreamVar("symbolRuns"), pb.getInteger(0)), runs);
}


class RunLengthSelector final: public pablo::PabloKernel {
public:
    RunLengthSelector(LLVMTypeSystemInterface & ts,
                      unsigned lo,
                      unsigned hi,
                      StreamSet * symbolRun, StreamSet * const lengthBixNum,
                      StreamSet * overflow,
                      StreamSet * selected);
protected:
    void generatePabloMethod() override;
    unsigned mLo;
    unsigned mHi;
};

RunLengthSelector::RunLengthSelector(LLVMTypeSystemInterface & ts,
                           unsigned lo,
                           unsigned hi,
                           StreamSet * symbolRun,
                           StreamSet * const lengthBixNum,
                           StreamSet * overflow,
                           StreamSet * selected)
: PabloKernel(ts, "RunLengthSelector" + std::to_string(lengthBixNum->getNumElements()) + "x1:" + std::to_string(lo) + "-" + std::to_string(lo),
              {Binding{"symbolRun", symbolRun, FixedRate(), LookAhead(1)},
                  Binding{"lengthBixNum", lengthBixNum},
                  Binding{"overflow", overflow}},
              {Binding{"selected", selected}}), mLo(lo), mHi(hi) { }

void RunLengthSelector::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * run = getInputStreamSet("symbolRun")[0];
    std::vector<PabloAST *> lengthBixNum = getInputStreamSet("lengthBixNum");
    PabloAST * overflow = getInputStreamSet("overflow")[0];
    PabloAST * runFinal = pb.createAnd(run, pb.createNot(pb.createLookahead(run, 1)));
    runFinal = pb.createAnd(runFinal, pb.createNot(overflow));
    Var * groupStreamVar = getOutputStreamVar("selected");
    // Run index codes count from 0 on the 2nd byte of a symbol.
    // So the length is 2 more than the bixnum.
    unsigned offset = 2;
    PabloAST * groupStream = pb.createAnd3(bnc.UGE(lengthBixNum, mLo - offset), bnc.ULE(lengthBixNum, mHi - offset), runFinal);
    pb.createAssign(pb.createExtract(groupStreamVar, pb.getInteger(0)), groupStream);
}

const unsigned hash_bits = 8;
const unsigned hash_entries = 1 << hash_bits;

class HashTable6 {
public:
    HashTable6() {
        for (unsigned i = 0; i < hash_entries; i++) {
            mTable[i] = {};
        }
    }
    
    void insert(uint8_t hash_code, const char * key_ptr) {
        uint64_t word_val = *(reinterpret_cast<const uint32_t *>(key_ptr));
        uint64_t word_val2 = *(reinterpret_cast<const uint16_t *>(key_ptr + 4));
        word_val = word_val + (word_val2 << 32);
        bool found = false;
        for (unsigned i = 0; i < mTable[hash_code].size(); i++) {
            if (mTable[hash_code][i] == word_val) {
                mCount[hash_code][i]++;
            }
            found = true;
        }
        if (not found) {
            mTable[hash_code].push_back(word_val);
            mCount[hash_code].push_back(1);
        }
    }
        
    void print() {
        for (unsigned h = 0; h < hash_entries; h++) {
            for (unsigned i = 0; i < mTable[h].size(); i++) {
                uint64_t val = mTable[h][i];
                char * key_ptr = reinterpret_cast<char *>(&val);
                unsigned key_count = mCount[h][i];
                std::cout << std::string(key_ptr, 6) << ": " << std::to_string(key_count) << " occurrences\n";
            }
        }
    }
private:
    std::vector<uint64_t> mTable[hash_entries];
    std::vector<unsigned> mCount[hash_entries];
};

HashTable6 T6;

typedef void (*HashDemoFunctionType)(uint32_t fd);


extern "C" void callback(const char * L6end_ptr, uint8_t hashval) {
    const char * L6_start_ptr = L6end_ptr - 5;
    //std::cout << std::string(L6_start_ptr, 6) << " " << std::to_string(hashval) << "\n";
    T6.insert(hashval, L6_start_ptr);
}

HashDemoFunctionType hashdemo_gen (CPUDriver & driver) {

    auto P = CreatePipeline(driver, Input<uint32_t>{"inputFileDecriptor"});

    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");

    // Source data
    StreamSet * const codeUnitStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, codeUnitStream);

    StreamSet * const u8basis = P.CreateStreamSet(8);
    P.CreateKernelCall<S2PKernel>(codeUnitStream, u8basis);
    SHOW_BYTES(codeUnitStream);
    SHOW_BIXNUM(u8basis);

    StreamSet * WordChars = P.CreateStreamSet(1);
    P.CreateKernelCall<WordMarkKernel>(u8basis, WordChars);
    SHOW_STREAM(WordChars);
    
    StreamSet * const SymbolRuns = P.CreateStreamSet(1);
    P.CreateKernelCall<ParseSymbols>(u8basis, WordChars, SymbolRuns);
    SHOW_STREAM(SymbolRuns);
    
    StreamSet * const runIndex = P.CreateStreamSet(4);
    StreamSet * const overflow = P.CreateStreamSet(1);
    P.CreateKernelCall<RunIndex>(SymbolRuns, runIndex, overflow);
    SHOW_BIXNUM(runIndex);
    SHOW_STREAM(overflow);
    
    StreamSet * const BixHashes = P.CreateStreamSet(hash_bits);
    P.CreateKernelCall<BixHash>(u8basis, SymbolRuns, BixHashes, 3);
    SHOW_BIXNUM(BixHashes);


    StreamSet * Lgth6symEnds = P.CreateStreamSet(1);
    P.CreateKernelCall<RunLengthSelector>(6, 6, SymbolRuns, runIndex, overflow, Lgth6symEnds);
    SHOW_STREAM(Lgth6symEnds);
    
    StreamSet * const L6_Hashes = P.CreateStreamSet(hash_bits);
    FilterByMask(P, Lgth6symEnds, BixHashes, L6_Hashes);
    SHOW_BIXNUM(L6_Hashes);
    
    StreamSet * const hashValues = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<P2SKernel>(L6_Hashes, hashValues);

    StreamSet * const scanIndices = P.CreateStreamSet(1, 64);
    P.CreateKernelCall<ScanIndexGenerator>(Lgth6symEnds, scanIndices);
    
    scan::Reader(P, driver, SCAN_CALLBACK(callback), codeUnitStream, scanIndices, { hashValues });
    
    return P.compile();
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&HashDemoOptions, &codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});
    CPUDriver driver("hashdemo");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
    } else {
        HashDemoFunctionType func = nullptr;
        func = hashdemo_gen(driver);
        func(fd);
        close(fd);
    }
    T6.print();
    return 0;
}
