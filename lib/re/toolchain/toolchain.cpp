/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/toolchain/toolchain.h>

#include <toolchain/toolchain.h>
#include <llvm/Support/CommandLine.h>

using namespace llvm;

namespace re {

cl::OptionCategory RE_Options("A. Regular Expression Interpretation", 
    "These options control regular expression parsing and interpretation");

static cl::bits<RE_PrintFlags>
    PrintOptions(cl::values(clEnumVal(ShowREs, "Show parsed regular expressions and transformations that change them"),
                            clEnumVal(ShowAllREs, "Print all regular expression passes")), cl::cat(codegen::JIT_InfoOptions));

static cl::bits<RE_AlgorithmFlags>
    AlgorithmOptions(cl::values(clEnumVal(DisableLog2BoundedRepetition, "disable log2 optimizations for bounded repetition of bytes"),
                              clEnumVal(DisableMatchStar, "disable MatchStar optimization")), cl::cat(codegen::CodeGenOptions));

static cl::opt<bool> UnicodeLevel2("U2", cl::desc("Enable Unicode Level matching under canonical and compatible (?K) equivalence."), 
    cl::Hidden, cl::cat(RE_Options));

bool LLVM_READONLY PrintOptionIsSet(RE_PrintFlags flag) {
    return PrintOptions.isSet(flag);
}

bool LLVM_READONLY AlgorithmOptionIsSet(RE_AlgorithmFlags flag) {
    return AlgorithmOptions.isSet(flag);
}

bool LLVM_READONLY UnicodeLevel2IsSet() {
    return UnicodeLevel2;
}

const int DefaultIfInsertionGap = 3;
int IfInsertionGap;
static cl::opt<int, true>
    IfInsertionGapOption("if-insertion-gap",  cl::location(IfInsertionGap), cl::init(DefaultIfInsertionGap),
                         cl::desc("minimum number of nonempty elements between inserted if short-circuit tests"),
                         cl::cat(codegen::CodeGenOptions));

std::string AnnotateWithREflags(std::string name) {
    if (re::AlgorithmOptionIsSet(re::DisableMatchStar)) {
        name += "-MatchStar";
    }
    if (re::AlgorithmOptionIsSet(re::DisableLog2BoundedRepetition)) {
        name += "-log2rep";
    }
    if (IfInsertionGap != DefaultIfInsertionGap) {
        name += "+ifGap="+std::to_string(IfInsertionGap);
    }
    return name;
}

} // namespace re
