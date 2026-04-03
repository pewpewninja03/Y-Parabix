/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */


#include <cstdio>
#include <vector>
#include <csv/csv_cmdline.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>

using namespace llvm;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html

namespace csv {
cl::OptionCategory CSV_Options("CSV Processing Options", "CSV Processing Options.");
bool HeaderSpecNamesFile;
static cl::opt<bool, true> HeaderSpecNamesFileOption("f", cl::location(HeaderSpecNamesFile), cl::desc("Interpret headers parameter as file name with header line"), cl::init(false), cl::cat(CSV_Options));
std::string HeaderSpec;
static cl::opt<std::string, true> HeaderSpecOption("headers", cl::location(HeaderSpec), cl::desc("CSV column headers (explicit string or filename"), cl::init(""), cl::cat(CSV_Options));
}

