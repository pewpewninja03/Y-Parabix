/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <string>
#include <llvm/Support/Compiler.h>

namespace llvm { namespace cl { class OptionCategory; } }

namespace re {

extern llvm::cl::OptionCategory RE_Options;

enum RE_PrintFlags {
    ShowREs, ShowAllREs
};
    
enum RE_AlgorithmFlags {
    DisableLog2BoundedRepetition, DisableIfHierarchy, DisableMatchStar
};
    
bool LLVM_READONLY PrintOptionIsSet(RE_PrintFlags flag);
bool LLVM_READONLY AlgorithmOptionIsSet(RE_AlgorithmFlags flag);
bool LLVM_READONLY UnicodeLevel2IsSet();

extern int IfInsertionGap;

std::string AnnotateWithREflags(std::string name);

}
