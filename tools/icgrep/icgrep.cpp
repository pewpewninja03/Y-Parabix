/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <cstdio>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/raw_ostream.h>
#include <re/adt/re_alt.h>
#include <re/adt/re_seq.h>
#include <re/adt/re_start.h>
#include <re/adt/re_end.h>
#include <re/adt/re_utility.h>
#include <re/parse/parser.h>
#include <re/toolchain/toolchain.h>
#include <grep/grep_engine.h>
#include "grep_interface.h"
#include <fstream>
#include <string>
#include <toolchain/toolchain.h>
#include <boost/filesystem.hpp>
#include <fileselect/file_select.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <llvm/ADT/STLExtras.h> // for make_unique
#include <kernel/pipeline/driver/cpudriver.h>
#ifdef ENABLE_PAPI
#include <util/papi_helper.hpp>
#endif

using namespace llvm;

static cl::list<std::string> inputFiles(cl::Positional, cl::desc("<regex> <input file ...>"), cl::OneOrMore);

static cl::opt<bool> ByteMode("enable-byte-mode", cl::desc("Process regular expressions in byte mode"));

static cl::opt<int> REsPerGroup("re-num", cl::desc("Number of regular expressions processed by each kernel."), cl::init(0));

static re::ModeFlagSet globalFlags = re::MULTILINE_MODE_FLAG;

re::RE * readRE() {

    if (argv::FileFlag != "") {
        std::ifstream regexFile(argv::FileFlag.c_str());
        std::string r;
        if (regexFile.is_open()) {
            while (std::getline(regexFile, r)) {
                argv::RegexpVector.push_back(r);
            }
            regexFile.close();
        }
    }

    // if there are no regexes specified through -e or -f, the first positional argument
    // must be a regex, not an input file.

    if (argv::RegexpVector.size() == 0) {
        argv::RegexpVector.push_back(inputFiles[0]);
        inputFiles.erase(inputFiles.begin());
    }
    if (argv::IgnoreCaseFlag) {
        globalFlags |= re::CASE_INSENSITIVE_MODE_FLAG;
    }

    std::vector<re::RE *> REs;
    for (unsigned i = 0; i < argv::RegexpVector.size(); i++) {
        re::RE * re_ast = re::RE_Parser::parse(argv::RegexpVector[i], globalFlags, argv::RegexpSyntax, ByteMode);
        REs.push_back(re_ast);
    }
    re::RE * fullRE = makeAlt(REs.begin(), REs.end());

    if (argv::WordRegexpFlag) {
        fullRE = re::makeSeq({re::makeWordBoundary(), fullRE, re::makeWordBoundary()});
    }
    if (argv::LineRegexpFlag) {
        fullRE = re::makeSeq({re::makeStart(), fullRE, re::makeEnd()});
    }

    return fullRE;
}

namespace fs = boost::filesystem;

int main(int argc, char *argv[]) {
    llvm_shutdown_obj shutdown;
    argv::InitializeCommandLineInterface(argc, argv);
    CPUDriver driver("icgrep");

    auto RE = readRE();

    const auto allFiles = argv::getFullFileList(driver, inputFiles);
    if (inputFiles.empty()) {
        argv::UseStdIn = true;
    } else if ((allFiles.size() > 1) && !argv::NoFilenameFlag) {
        argv::WithFilenameFlag = true;
    }

    std::unique_ptr<grep::GrepEngine> grep;
    switch (argv::Mode) {
        case argv::NormalMode:
            grep = std::make_unique<grep::EmitMatchesEngine>(driver);
            if (argv::MaxCountFlag) grep->setMaxCount(argv::MaxCountFlag);
            if (argv::WithFilenameFlag) grep->showFileNames();
            if (argv::LineNumberFlag) grep->showLineNumbers();
            if (argv::InitialTabFlag) grep->setInitialTab();
            if ((argv::ColorFlag == argv::alwaysColor) ||
                ((argv::ColorFlag == argv::autoColor) && isatty(STDOUT_FILENO))) {
                grep->setColoring();
            }
           break;
        case argv::CountOnly:
        case argv::CountAll:
            grep = std::make_unique<grep::CountOnlyEngine>(driver, argv::Mode == argv::CountAll);
            if (argv::WithFilenameFlag) grep->showFileNames();
            if (argv::MaxCountFlag) grep->setMaxCount(argv::MaxCountFlag);
           break;
        case argv::FilesWithMatch:
        case argv::FilesWithoutMatch:
            grep = std::make_unique<grep::MatchOnlyEngine>(driver, argv::Mode == argv::FilesWithMatch, argv::NullFlag);
            break;
        case argv::QuietMode:
           grep = std::make_unique<grep::QuietModeEngine>(driver);
            break;
        default: llvm_unreachable("Invalid grep mode!");
    }
    if (argv::IgnoreCaseFlag) grep->setCaseInsensitive();
    if (argv::InvertMatchFlag) grep->setInvertMatches();
    if (argv::UnicodeLinesFlag) {
        grep->setRecordBreak(grep::GrepRecordBreakKind::Unicode);
    } else if (argv::NullDataFlag) {
        grep->setRecordBreak(grep::GrepRecordBreakKind::Null);
    } else {
        grep->setRecordBreak(grep::GrepRecordBreakKind::LF);
    }
    grep->setContextLines(argv::BeforeContext, argv::AfterContext);

    grep->setStdinLabel(argv::LabelFlag);
    if (argv::UseStdIn) grep->setGrepStdIn();
    if (argv::NoMessagesFlag) grep->suppressFileMessages();
    if (argv::MmapFlag) grep->setPreferMMap();
    grep->setBinaryFilesOption(argv::BinaryFilesFlag);
    grep->initFileResult(allFiles); // unnecessary copy!
    grep->initRE(RE);
    grep->grepCodeGen();
    #ifdef REPORT_PAPI_TESTS
    papi::PapiCounter<4> jitExecution{{PAPI_L3_TCM, PAPI_L3_TCA, PAPI_TOT_INS, PAPI_TOT_CYC}};
    jitExecution.start();
    #endif
    const bool matchFound = grep->searchAllFiles();
    #ifdef REPORT_PAPI_TESTS
    jitExecution.stop();
    jitExecution.write(std::cerr);
    #endif
    return matchFound ? argv::MatchFoundExitCode : argv::MatchNotFoundExitCode;
}
