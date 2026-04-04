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
namespace llvm { namespace cl { class OptionCategory; } }

namespace csv {

extern llvm::cl::OptionCategory CSV_Options;

// input file parameter
extern std::string inputFile;

// header information
extern bool HeaderSpecNamesFile;
extern std::string HeaderSpec;

// delimiters
extern char32_t FieldDelimiter;
extern char32_t QuoteChar;

std::vector<std::string> get_CSV_headers();

// Specifying columns
extern std::vector<std::string> Columns;
extern bool ZeroIndexing;

std::vector<unsigned> getColumnArgs(std::vector<std::string> & headers);

void InitializeCommandLineInterface(int argc, char *argv[]);


//
// Command line exit codes.
enum ExitCode {
    SuccessExitCode = 0, 
    InternalFailureCode = 2,      // Fatal error code due to program logic or system problem.
    UsageErrorCode = 3            // Error in command line parameters.
};

}
