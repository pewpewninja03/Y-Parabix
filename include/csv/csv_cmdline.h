/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 *
 *  This file defines the command line options interface for Parabix csv tools.
 *
 */
#pragma once
 
#include <string>
#include <vector>
//#include <llvm/Support/CommandLine.h>
namespace llvm { namespace cl { class OptionCategory; } }

namespace csv {

extern llvm::cl::OptionCategory CSV_Options;

extern bool HeaderSpecNamesFile;
extern std::string HeaderSpec;

    
//void InitializeCommandLineInterface(int argc, char *argv[]);

}
