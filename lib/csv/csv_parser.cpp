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

void CSV_Lexer(PipelineBuilder & P, StreamSet * source, StreamSet * csvCCs) {
    std::vector<re::CC *> csv_marks(4);
    csv_marks[markLF] = re::makeCC(charLF);
    csv_marks[markCR] = re::makeCC(charCR);
    csv_marks[markDQ] = re::makeCC(csv::QuoteChar);
    csv_marks[markComma] = re::makeCC(csv::FieldDelimiter);
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

class SelectMultiField : public PabloKernel {
public:
    SelectMultiField(LLVMTypeSystemInterface & ts,
                              StreamSet * Record_separators,
                              StreamSet * Field_separators,
                              StreamSet * toKeep,
                              const std::vector<unsigned> & columnNos,
                              bool forCut);
protected:
    std::string colNoString(const std::vector<unsigned> & cols, bool forCut);
    void generatePabloMethod() override;
    const std::vector<unsigned> & mColumnNos;
    bool mForCut;
};

SelectMultiField::SelectMultiField(LLVMTypeSystemInterface & ts,
                                   StreamSet * Record_separators,
                                   StreamSet * Field_separators,
                                   StreamSet * toKeep,
                                   const std::vector<unsigned> & columnNos,
                                   bool forCut)
: PabloKernel(ts, "SelectMultiField_" + colNoString(columnNos, forCut),
  {Binding{"Record_separators", Record_separators},
   Binding{"Field_separators", Field_separators}},
  {Binding{"toKeep", toKeep}}), mColumnNos(columnNos), mForCut(forCut) {
}

std::string SelectMultiField::colNoString(const std::vector<unsigned> & columnNos, bool forCut) {
    std::stringstream ss;
    ss << columnNos[0];
    for (unsigned i = 1; i < columnNos.size(); i++) {
        ss << "," << columnNos[i];
    }
    if (forCut) {
        ss << "_c";
    }
    return ss.str();
}

void SelectMultiField::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * Record_separators = pb.createExtract(getInputStreamVar("Record_separators"), pb.getInteger(0));
    PabloAST * Field_separators = pb.createExtract(getInputStreamVar("Field_separators"), pb.getInteger(0));
    PabloAST * recordStarts = pb.createNot(pb.createAdvance(pb.createNot(Record_separators), 1));
    PabloAST * fieldStarts = pb.createNot(pb.createAdvance(pb.createNot(Field_separators), 1));
    PabloAST * columnMark = recordStarts;
    PabloAST * toKeep = pb.createZeroes();
    unsigned nextCol = 0;
    for (unsigned i = 0; i < mColumnNos.size(); i++) {
        if (mColumnNos[i] > nextCol) {
            columnMark = pb.createIndexedAdvance(columnMark, fieldStarts, mColumnNos[i] - nextCol);
        }
        PabloAST * columnFollow = pb.createScanTo(columnMark, Field_separators);
        PabloAST * columnMask  = pb.createIntrinsicCall(pablo::Intrinsic::SpanUpTo, {columnMark, columnFollow});
        toKeep = pb.createOr(toKeep, columnMask);
        nextCol = mColumnNos[i] + 1;
        columnMark = pb.createAdvance(columnFollow, 1);
        if (mForCut && (i == mColumnNos.size() - 1)) {
            toKeep = pb.createOr(toKeep, Record_separators);
        } else {
            toKeep = pb.createOr(toKeep, columnFollow);
        }
    }
    pb.createAssign(pb.createExtract(getOutputStreamVar("toKeep"), pb.getInteger(0)), pb.createInFile(toKeep));
}

void ColumnSelectionMask(PipelineBuilder & P, StreamSet * Record_separators, StreamSet * Field_separators,
                         StreamSet * toKeep, const std::vector<unsigned> & columnNos, bool forCut) {
    P.CreateKernelCall<SelectMultiField>(Record_separators, Field_separators, toKeep, columnNos, forCut);
}

class EmptyFieldsKernel : public PabloKernel {
public:
    EmptyFieldsKernel(LLVMTypeSystemInterface & ts, StreamSet * csvCCs, StreamSet * fieldSeparators,
                      StreamSet * EmptyFields);
protected:
    void generatePabloMethod() override;
};

EmptyFieldsKernel::EmptyFieldsKernel(LLVMTypeSystemInterface & ts, StreamSet * csvCCs, StreamSet * fieldSeparators,
                                     StreamSet * EmptyMarks)
: PabloKernel(ts, "EmptyFieldsKernel" + csvCCs->shapeString(),
  {Binding{"csvCCs", csvCCs},
   Binding{"fieldSeparators", fieldSeparators, FixedRate(), LookAhead(1)}},
  {Binding{"EmptyMarks", EmptyMarks}})  {
}

void EmptyFieldsKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvCCs");
    PabloAST * fieldSeparators = pb.createExtract(getInputStreamVar("fieldSeparators"), pb.getInteger(0));
    PabloAST * fsNext = pb.createAdvance(fieldSeparators, 1);
    PabloAST * quotedStart = pb.createAnd(fsNext, csvMarks[csv::markDQ]);
    PabloAST * quoted2 = pb.createAnd(pb.createAdvance(quotedStart, 1), csvMarks[csv::markDQ]);
    PabloAST * empties = pb.createAnd(pb.createOr(fsNext, pb.createAdvance(quoted2, 1)), fieldSeparators);
    pb.createAssign(pb.createExtract(getOutputStreamVar("EmptyMarks"), pb.getInteger(0)), empties);
}

void GetEmptyFields(PipelineBuilder & P, StreamSet * csvCCs, StreamSet * fieldSeparators,
                    StreamSet * EmptyFieldMarks) {
    P.CreateKernelCall<EmptyFieldsKernel>(csvCCs, fieldSeparators, EmptyFieldMarks);
}
}
