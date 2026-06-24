/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include "editdscan_kernel.h"
#include "pattern_compiler.h"
#include "editd_cpu_kernel.h"
#include <string>
#include <iostream>
#include <fstream>
#include <toolchain/toolchain.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/idisa_target.h>
#include <kernel/core/streamset.h>
#include <kernel/io/source_kernel.h>
#include <kernel/streamutils/streams_merge.h>
#include <pablo/pablo_compiler.h>
#include <pablo/pablo_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/cc/cc_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <mutex>
#include <kernel/pipeline/program_builder.h>
#include <util/aligned_allocator.h>
#include <kernel/core/streamsetptr.h>
#ifdef ENABLE_PAPI
#include <util/papi_helper.hpp>
#endif

// #define USE_MULTI_EDITD_MERGE_GROUPS

using namespace llvm;

static cl::list<std::string> inputFiles(cl::Positional, cl::desc("<regex> <input file ...>"), cl::OneOrMore);

static cl::list<std::string> pattVector("e", cl::desc("pattern"), cl::ZeroOrMore);
static cl::opt<std::string> PatternFilename("f", cl::desc("Take patterns (one per line) from a file"), cl::value_desc("regex file"), cl::init(""));

static cl::opt<bool> CaseInsensitive("i", cl::desc("Ignore case distinctions in the pattern and the file."));

static cl::opt<int> editDistance("edit-dist", cl::desc("Edit Distance Value"), cl::init(2));
static cl::opt<int> optPosition("opt-pos", cl::desc("Optimize position"), cl::init(0));
static cl::opt<int> stepSize("step-size", cl::desc("Step Size"), cl::init(3));
static cl::opt<int> prefixLen("prefix", cl::desc("Prefix length"), cl::init(3));
static cl::opt<unsigned> groupSize("groupPatterns", cl::desc("Number of pattern segments per group."), cl::init(1));
static cl::opt<bool> ShowPositions("display", cl::desc("Display the match positions."), cl::init(false));

static cl::opt<bool> UsePhases("phased", cl::desc("Use phased pipeline stages instead of multiple pipelines."), cl::init(false));

static cl::opt<int> Threads("threads", cl::desc("Total number of threads."), cl::init(1));

static cl::opt<bool> MultiEditdKernels("enable-multieditd-kernels", cl::desc("Construct multiple editd kernels in one pipeline."));
#ifdef USE_MULTI_EDITD_MERGE_GROUPS
static cl::opt<unsigned> MergeGroupSize("merge-group-size", cl::desc("Number of editd kernels executed before merging results"), cl::init(4));
#endif

static cl::opt<bool> EditdIndexPatternKernels("enable-index-kernels", cl::desc("Use pattern index method."));

using namespace kernel;
using namespace pablo;

struct matchPosition
{
    size_t pos;
    size_t dist;
};

std::vector<struct matchPosition> matchList;
std::vector<std::vector<std::string>> pattGroups;

void run_second_filter(int pattern_segs, int total_len, float errRate){

    if(matchList.empty()) return;

    //Sort match position
    bool exchanged = true;
    while(exchanged){
        exchanged = false;
        for (unsigned i=0; i<matchList.size()-1; i++){
            if(matchList[i].pos > matchList[i+1].pos){
                size_t tmp_pos = matchList[i].pos;
                size_t tmp_dist = matchList[i].dist;
                matchList[i].pos = matchList[i+1].pos;
                matchList[i].dist = matchList[i+1].dist;
                matchList[i+1].pos = tmp_pos;
                matchList[i+1].dist = tmp_dist;
                exchanged = true;
            }
        }
    }

    //remove the duplicates
    bool cleared = true;
    while(cleared){
        cleared = false;
        for (unsigned i=0; i<matchList.size()-1; i++){
            if(matchList[i].pos == matchList[i+1].pos && matchList[i].dist == matchList[i+1].dist){
                matchList.erase(matchList.begin() + i);
                cleared = true;
            }
        }
    }

    std::cout << "pattern_segs = " << pattern_segs << ", total_len = " << total_len << std::endl;

    int v = pattern_segs * (editDistance+1) - total_len * errRate;

    int startPos = matchList[0].pos;
    int sum = matchList[0].dist;
    int curIdx = 0;
    unsigned i = 0;
    int count = 0;
    while (i < matchList.size()){
        if(matchList[i].pos - startPos < total_len * (errRate+1)){
            sum += matchList[i].dist;
            i++;
        }
        else{
            if(sum > v) count++;
            sum -= matchList[curIdx].dist;
            curIdx++;
            startPos = matchList[curIdx].pos;
        }
    }

    std::cout << "total candidate from the first filter is " << matchList.size() << std::endl;
    std::cout << "total candidate from the second filter is " << count << std::endl;
}

void get_editd_pattern(int & pattern_segs, int & total_len) {

    if (PatternFilename != "") {
        std::ifstream pattFile(PatternFilename.c_str());
        std::string r;
        if (pattFile.is_open()) {
            while (std::getline(pattFile, r)) {
                pattVector.push_back(r);
                pattern_segs ++;
                total_len += r.size();
            }
            std::sort(pattVector.begin(), pattVector.end());
            unsigned i = 0;
            while(i < pattVector.size()){
                std::vector<std::string> pattGroup;
                std::string prefix = pattVector[i].substr(0, prefixLen);
                while(i < pattVector.size() && pattVector[i].substr(0, prefixLen) == prefix){
                    pattGroup.push_back(pattVector[i]);
                    i++;
                }
                pattGroups.push_back(pattGroup);
            }
            pattFile.close();
        }
        codegen::GroupNum = pattVector.size()/groupSize;
    }

    // if there are no regexes specified through -e or -f, the first positional argument
    // must be a regex, not an input file.

    if (pattVector.size() == 0 && inputFiles.size() > 1) {
        pattVector.push_back(inputFiles[0]);
        inputFiles.erase(inputFiles.begin());
    }
}

typedef void (*preprocessFunctionType)(StreamSetPtr & chStream, const uint32_t fd);

class PreprocessKernel final: public pablo::PabloKernel {
public:
    PreprocessKernel(LLVMTypeSystemInterface & ts, StreamSet * BasisBits, StreamSet * CCResults);
protected:
    void generatePabloMethod() override;
};

PreprocessKernel::PreprocessKernel(LLVMTypeSystemInterface & ts, StreamSet * BasisBits, StreamSet * CCResults)
: PabloKernel(ts, "editd_preprocess", {{"basis", BasisBits}}, {{"pat", CCResults}}) {

}

void PreprocessKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    cc::Parabix_CC_Compiler_Builder ccc(getInputStreamSet("basis"));
    PabloAST * A = ccc.compileCC(re::makeCC(re::makeCC(0x41), re::makeCC(0x61)), pb);
    PabloAST * C = ccc.compileCC(re::makeCC(re::makeCC(0x43), re::makeCC(0x63)), pb);
    PabloAST * T = ccc.compileCC(re::makeCC(re::makeCC(0x54), re::makeCC(0x74)), pb);
    PabloAST * G = ccc.compileCC(re::makeCC(re::makeCC(0x47), re::makeCC(0x67)), pb);
    Var * const pat = getOutputStreamVar("pat");
    pb.createAssign(pb.createExtract(pat, 0), A);
    pb.createAssign(pb.createExtract(pat, 1), C);
    pb.createAssign(pb.createExtract(pat, 2), T);
    pb.createAssign(pb.createExtract(pat, 3), G);
}

void preprocessPipeline(ProgramBuilder & P, StreamSet * const CCResults) {
    Scalar * const fileDescriptor = P.getInputScalar("fileDescriptor");
    StreamSet * const ByteStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
    StreamSet * const BasisBits = P.CreateStreamSet(8);
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    P.CreateKernelCall<PreprocessKernel>(BasisBits, CCResults);
}

preprocessFunctionType preprocessPipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver, Output<streamset_t>("chStream", 4, 1, ReturnedBuffer(1)), Input<uint32_t>("fileDescriptor"));
    StreamSet * const CCResults = P.getOutputStreamSet("chStream");
    preprocessPipeline(P, CCResults);
    return P.compile();
}

StreamSetPtr preprocess(preprocessFunctionType preprocess) {
    const auto fd = open(inputFiles[0].c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        std::cerr << "Error: cannot open " << inputFiles[0] << " for processing.\n";
        exit(-1);
    }
    StreamSetPtr chStream;
    preprocess(chStream, fd);
    close(fd);
    return chStream;
}

template<typename FuncType, typename ... Args>
void runPhasedEditPipeline(FuncType editdFunc, Args... args) {
    const auto fd = open(inputFiles[0].c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        std::cerr << "Error: cannot open " << inputFiles[0] << " for processing.\n";
        exit(-1);
    }
    editdFunc(fd, args...);
    close(fd);
}

LLVM_READNONE std::string createName(const std::vector<std::string> & patterns) {
    std::string name;
    raw_string_ostream out(name);
    for(const auto & pat : patterns) {
        out << pat;
    }
    out << std::to_string(editDistance);
    out.flush();
    return name;
}

class PatternKernel final : public pablo::PabloKernel {
public:
    PatternKernel(LLVMTypeSystemInterface & ts, const std::vector<std::string> & patterns, StreamSet * pat, StreamSet * E);
    StringRef getSignature() const override {
        return mSignature;
    }
    bool hasSignature() const override { return true; }
protected:
    void generatePabloMethod() override;
private:
    const std::vector<std::string> & mPatterns;
    const std::string mSignature;
};

PatternKernel::PatternKernel(LLVMTypeSystemInterface & ts, const std::vector<std::string> & patterns, StreamSet * pat, StreamSet * E)
: PabloKernel(ts, "Editd_pattern_" + getStringHash(createName(patterns)),
{{"pat", pat}},
{{"E", E}})
, mPatterns(patterns)
, mSignature(createName(patterns)) {

}

void PatternKernel::generatePabloMethod() {
    PabloBuilder entry(getEntryScope());
    Var * const pat = getInputStreamVar("pat");
    PabloAST * basisBits[4];
    basisBits[0] = entry.createExtract(pat, 0, "A");
    basisBits[1] = entry.createExtract(pat, 1, "C");
    basisBits[2] = entry.createExtract(pat, 2, "T");
    basisBits[3] = entry.createExtract(pat, 3, "G");
    re::Pattern_Compiler pattern_compiler(*this);
    if (optPosition == 0) optPosition = editDistance + 6;
    pattern_compiler.compile(mPatterns, entry, basisBits, editDistance, optPosition, stepSize);
}

void wrapped_report_pos(size_t match_pos, int dist) {
    struct matchPosition curMatch;
    curMatch.pos = match_pos;
    curMatch.dist = dist;
    matchList.push_back(curMatch);
    if (ShowPositions) {
        std::cout << "pos: " << match_pos << ", dist:" << dist << "\n";
    }
}

typedef void (*editdFunctionType)(const StreamSetPtr & chStream);

void editdPipeline(ProgramBuilder & P, const std::vector<std::string> & patterns, StreamSet * const ChStream) {
    StreamSet * const MatchResults = P.CreateStreamSet(editDistance + 1);
    P.CreateKernelFamilyCall<PatternKernel>(patterns, ChStream, MatchResults);
    Kernel * const scan = P.CreateKernelCall<editdScanKernel>(MatchResults);
    scan->link("wrapped_report_pos", wrapped_report_pos);
}

editdFunctionType editdPipeline(CPUDriver & driver, const std::vector<std::string> & patterns) {
    auto P = CreatePipeline(driver, Input<streamset_t>("chStream", 4, 1));
    StreamSet * const ChStream = P.getInputStreamSet(0);
    editdPipeline(P, patterns, ChStream);
    return P.compile();
}

typedef void (*multiEditdFunctionType)(const uint32_t fd);

void multiEditdPipeline(ProgramBuilder & P) {

    Scalar * const fileDescriptor = P.getInputScalar("fileDescriptor");

    StreamSet * const ByteStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);

    StreamSet * const BasisBits = P.CreateStreamSet(8);
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);

    StreamSet * const ChStream = P.CreateStreamSet(4);
    P.CreateKernelCall<PreprocessKernel>(BasisBits, ChStream);

    const auto n = pattGroups.size();
    if (n == 0) {
        report_fatal_error("no patterns found");
        exit(-2);
    }
    std::vector<StreamSet *> MatchResults(n);
    for(unsigned i = 0; i < n; ++i){
        MatchResults[i] = P.CreateStreamSet(editDistance + 1);
        P.CreateKernelFamilyCall<PatternKernel>(pattGroups[i], ChStream, MatchResults[i]);
    }

    StreamSet * finalResults = MatchResults[0];
    if (n > 1) {
        #ifdef USE_MULTI_EDITD_MERGE_GROUPS
        const unsigned m = MergeGroupSize.getValue();
        if (m < 2) {
            report_fatal_error("merge-group-size cannot be less than 2");
        }
        std::vector<StreamSet *> mergeGroup;
        mergeGroup.reserve(m);
        assert (MatchResults[0]);
        mergeGroup.push_back(MatchResults[0]);
        for (unsigned i = 1; i < n; ++i) {
            if (mergeGroup.size() >= m) {
                StreamSet * const result = P->CreateStreamSet(editDistance + 1);
                P.CreateKernelCall<StreamsMerge>(mergeGroup, result);
                mergeGroup.clear();
                assert (result);
                mergeGroup.push_back(result);
            }
            assert (mergeGroup.size() < m);
            assert (MatchResults[i]);
            mergeGroup.push_back(MatchResults[i]);
        }
        assert (mergeGroup.size() > 1);
        finalResults = P->CreateStreamSet(editDistance + 1);
        assert (finalResults);
        P.CreateKernelCall<StreamsMerge>(mergeGroup, finalResults);
        #else
        finalResults = P.CreateStreamSet(editDistance + 1);
        P.CreateKernelCall<StreamsMerge>(MatchResults, finalResults);
        #endif
    }
    Kernel * const scan = P.CreateKernelCall<editdScanKernel>(finalResults);
    scan->link("wrapped_report_pos", wrapped_report_pos);
}

multiEditdFunctionType multiEditdPipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver, Input<uint32_t>("fileDescriptor"));
    multiEditdPipeline(P);
    return P.compile();
}

typedef void (*editdIndexFunctionType)(const StreamSetPtr & byteData, const char * pattern);

void editdIndexPatternPipeline(ProgramBuilder & P, unsigned patternLen, StreamSet * ChStream) {
    Scalar * const pattStream = P.getInputScalar("pattStream");
    StreamSet * const MatchResults = P.CreateStreamSet(editDistance + 1);
    P.CreateKernelCall<editdCPUKernel>(editDistance, patternLen, groupSize, pattStream, ChStream, MatchResults);
    Kernel * const scan = P.CreateKernelCall<editdScanKernel>(MatchResults);
    scan->link("wrapped_report_pos", wrapped_report_pos);
}


editdIndexFunctionType editdIndexPatternPipeline(CPUDriver & driver, unsigned patternLen) {
    auto P = CreatePipeline(driver, Input<streamset_t>("chStream", 4, 1), Input<const char*>("pattStream"));
    StreamSet * const ChStream = P.getInputStreamSet(0);
    editdIndexPatternPipeline(P, patternLen, ChStream);
    return P.compile();
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv);
    int pattern_segs = 0;
    int total_len = 0;

    get_editd_pattern(pattern_segs, total_len);

    CPUDriver driver("editd");
    if (MultiEditdKernels) {
        auto editd = multiEditdPipeline(driver);
        const auto & fileName = inputFiles[0];
        const int fd = open(inputFiles[0].c_str(), O_RDONLY);
        if (LLVM_UNLIKELY(fd == -1)) {
            std::cerr << "Error: cannot open " << fileName << " for processing. Skipped.\n";
            exit(-1);
        }
        #ifdef REPORT_PAPI_TESTS
        papi::PapiCounter<4> jitExecution{{PAPI_L3_TCM, PAPI_L3_TCA, PAPI_TOT_INS, PAPI_TOT_CYC}};
        jitExecution.start();
        #endif
        editd(fd);
        #ifdef REPORT_PAPI_TESTS
        jitExecution.stop();
        jitExecution.write(std::cerr);
        std::cerr << std::flush;
        #endif
        close(fd);
        #ifndef REPORT_PAPI_TESTS
        run_second_filter(pattern_segs, total_len, 0.15);
        #endif
        return 0;
    }

    if (UsePhases) {
        if (pattVector.size() == 1) {
            auto P = CreatePipeline(driver, Input<uint32_t>("fileDescriptor"));
            StreamSet * const CCResults = P.CreateStreamSet(4, 1);
            preprocessPipeline(P, CCResults);
            P.InsertPhaseBoundary();
            editdPipeline(P, pattVector, CCResults);
            runPhasedEditPipeline(P.compile());
            std::cout << "total matches is " << matchList.size() << std::endl;
        } else if (EditdIndexPatternKernels) {
            auto P = CreatePipeline(driver, Input<uint32_t>("fileDescriptor"), Input<const char*>("pattStream"));
            StreamSet * const CCResults = P.CreateStreamSet(4, 1);
            preprocessPipeline(P, CCResults);
            P.InsertPhaseBoundary();
            editdIndexPatternPipeline(P, pattVector[0].length(), CCResults);
            auto func = P.compile();
            const unsigned gs = groupSize;
            for(unsigned i=0; i< pattVector.size(); i += gs){
                SmallVector<char, 1024> pattern;
                for (unsigned j=0; j < gs; j++){
                    const auto & str = pattVector[i + j];
                    pattern.append(str.begin(), str.end());
                }
                runPhasedEditPipeline(func, pattern.data());
            }
        }
        else {
            auto P = CreatePipeline(driver, Input<uint32_t>("fileDescriptor"));
            StreamSet * const CCResults = P.CreateStreamSet(4, 1);
            preprocessPipeline(P, CCResults);
            for(unsigned i=0; i< pattGroups.size(); i++){
                P.InsertPhaseBoundary();
                editdPipeline(P, pattGroups[i], CCResults);
            }
            runPhasedEditPipeline(P.compile());
        }

    } else {


        auto preprocess_ptr = preprocessPipeline(driver);
        auto chStream = preprocess(preprocess_ptr);

        if (pattVector.size() == 1) {
            auto editd = editdPipeline(driver, pattVector);
            editd(chStream);
            std::cout << "total matches is " << matchList.size() << std::endl;
        } else if (EditdIndexPatternKernels) {
            auto editd_ptr = editdIndexPatternPipeline(driver, pattVector[0].length());
            const unsigned gs = groupSize;
            for(unsigned i=0; i< pattVector.size(); i += gs){
                SmallVector<char, 1024> pattern;
                for (unsigned j=0; j < gs; j++){
                    const auto & str = pattVector[i + j];
                    pattern.append(str.begin(), str.end());
                }
                editd_ptr(chStream, pattern.data());
            }
        }
        else {
            for(unsigned i=0; i< pattGroups.size(); i++){
                auto editd = editdPipeline(driver, pattGroups[i]);
                editd(chStream);
            }
        }

        free(chStream.data<8>());
    }




    run_second_filter(pattern_segs, total_len, 0.15);

    return 0;
}
