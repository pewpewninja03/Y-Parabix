#include <csv/csv_cmdline.h>
#include <csv/csv_parser.h>
#include <kernel/unicode/charclasses.h>
#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pe_ones.h>
#include <pablo/pablo_toolchain.h>
#include <pablo/bixnum/bixnum.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>

using namespace kernel;
using namespace pablo;

namespace csv {
char charLF = 0xA;
char charCR = 0xD;

void CSV_Lexer(PipelineBuilder & P, codepoint_t delimiter, codepoint_t quote, StreamSet * source, StreamSet * csvCCs) {
    std::vector<re::CC *> csv_marks(4);
    csv_marks[markLF] = re::makeCC(charLF);
    csv_marks[markCR] = re::makeCC(charCR);
    csv_marks[markDQ] = re::makeCC(quote);
    csv_marks[markComma] = re::makeCC(delimiter);
    P.CreateKernelCall<CharClassesKernel>(csv_marks, source, csvCCs);
}

class CSVparser : public PabloKernel {
public:
    CSVparser(LLVMTypeSystemInterface & ts, StreamSet * csvCCs, StreamSet * recordSeparators, StreamSet * fieldSeparators, StreamSet * quoteEscape)
        : PabloKernel(ts, "CSVparser",
                      {Binding{"csvCCs", csvCCs, FixedRate(), LookAhead(1)}},
                      {Binding{"recordSeparators", recordSeparators},
                       Binding{"fieldSeparators", fieldSeparators},
                       Binding{"quoteEscape", quoteEscape}}) {}
protected:
    void generatePabloMethod() override;
};

void CSVparser::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvCCs");
    PabloAST * dquote = csvMarks[markDQ];
    PabloAST * dquote_odd = pb.createEveryNth(dquote, pb.getInteger(2));
    PabloAST * dquote_even = pb.createXor(dquote, dquote_odd);
    PabloAST * quote_escape = pb.createAnd(dquote_even, pb.createLookahead(dquote, 1));
    PabloAST * escaped_quote = pb.createAdvance(quote_escape, 1);
    PabloAST * start_dquote = pb.createXor(dquote_odd, escaped_quote);
    PabloAST * end_dquote = pb.createXor(dquote_even, quote_escape);
    PabloAST * quoted_data = pb.createIntrinsicCall(pablo::Intrinsic::InclusiveSpan, {start_dquote, end_dquote});
    PabloAST * unquoted = pb.createNot(quoted_data);
    PabloAST * recordMarks = pb.createAnd(csvMarks[markLF], unquoted);
    PabloAST * fieldMarks = pb.createOr(pb.createAnd(csvMarks[markComma], unquoted), recordMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("recordSeparators"), pb.getInteger(0)), recordMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("fieldSeparators"), pb.getInteger(0)), fieldMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("quoteEscape"), pb.getInteger(0)), quote_escape);
}

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)

void ParseCSV(PipelineBuilder & P, StreamSet * csvCCs,
              StreamSet * recordSeparators, StreamSet * fieldSeparators, StreamSet * quoteEscape) {
    P.CreateKernelCall<CSVparser>(csvCCs, recordSeparators, fieldSeparators, quoteEscape);
    SHOW_STREAM(recordSeparators);
    SHOW_STREAM(fieldSeparators);
    SHOW_STREAM(quoteEscape);
}
}
