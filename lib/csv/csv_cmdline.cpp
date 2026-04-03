/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */


#include <cstdio>
#include <vector>
#include <fstream>
#include <csv/csv_cmdline.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/raw_ostream.h>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

using namespace llvm;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html

namespace csv {

cl::OptionCategory CSV_Options("CSV Processing Options", "CSV Processing Options.");


std::string inputFile;
static cl::opt<std::string, true> inputFileOption(cl::location(inputFile), cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(csv::CSV_Options));

// Although CSV files normally have column headers as the first line, 
// we can provide them directly or from a file.
bool HeaderSpecNamesFile;
std::string HeaderSpec;
static cl::opt<bool, true> HeaderSpecNamesFileOption("f", cl::location(HeaderSpecNamesFile), cl::desc("Interpret headers parameter as file name with header line"), cl::init(false), cl::cat(CSV_Options));
static cl::opt<std::string, true> HeaderSpecOption("headers", cl::location(HeaderSpec), cl::desc("CSV column headers (explicit string or filename"), cl::init(""), cl::cat(CSV_Options));


std::vector<std::string> parse_CSV_headers(std::string headerString) {
    std::vector<std::string> headers;
    boost::algorithm::split(headers, headerString, [] (char c) {return (c == ',');});
    for (unsigned i = 0; i < headers.size(); i++) {
        boost::algorithm::trim(headers[i]);
    }
    return headers;
}

std::vector<std::string> read_CSV_headers(std::string filename) {
    std::vector<std::string> headers;
    std::ifstream headerFile(filename.c_str());
    std::string line1;
    if (headerFile.is_open()) {
        std::getline(headerFile, line1);
        headerFile.close();
        headers = parse_CSV_headers(line1);
    } else {
        llvm::report_fatal_error(llvm::StringRef("Cannot open ") + filename);
    }
    return headers;
}

const unsigned MaxHeaderSize = 24;

std::vector<std::string> get_CSV_headers() {
	std::vector<std::string> headers;
    if (csv::HeaderSpec == "") {
        headers = csv::read_CSV_headers(csv::inputFile);
    } else if (csv::HeaderSpecNamesFile) {
        headers = csv::read_CSV_headers(csv::HeaderSpec);
    } else {
        headers = csv::parse_CSV_headers(csv::HeaderSpec);
    }
    for (auto & s : headers) {
        if (s.size() > MaxHeaderSize) {
            s = s.substr(0, MaxHeaderSize);
        }
    }
    return headers;
}

//
// Handler for errors reported through llvm::report_fatal_error.  Report
// and signal error the InternalFailure exit code.
//
static void csv_error_handler(void *UserData,
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(14, 0, 0)
							  const std::string &Message,
#else
							  const char * Message,
#endif
							  bool GenCrashDiag) {
    // Modified from LLVM's internal report_fatal_error logic.
    SmallVector<char, 64> Buffer;
    raw_svector_ostream OS(Buffer);
    OS << "Parabix csv error: " << Message << "\n";
    const auto MessageStr = OS.str();
    ssize_t written = ::write(2, MessageStr.data(), MessageStr.size());
    (void)written; // If something went wrong, we deliberately just give up.
    // Run the interrupt handlers to make sure any special cleanups get done, in
    // particular that we remove files registered with RemoveFileOnSignal.
    llvm::sys::RunInterruptHandlers();
    exit(InternalFailureCode);
}

void InitializeCommandLineInterface(int argc, char *argv[]) {
    llvm::install_fatal_error_handler(&csv_error_handler);
    codegen::ParseCommandLineOptions(argc, argv, {&CSV_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
}

}
