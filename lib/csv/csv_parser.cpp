#include <csv/csv_cmdline.h>
#include <csv/csv_parser.h>
#include <kernel/unicode/charclasses.h>
#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pe_ones.h>
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

//
//  Parse Quoted data within CSV files, based on the convention that
//  two quote marks in a row represent an escape sequence for an actual
//  quote character within the quoted string. 
//  Return a streamset consisting of two streams:
//  (a) a marker stream identifying the contents of each quoted string.
//  (b) a marker stream marking each quote escape character (first of
//      each pair of quotes within a quoted string).
class CSV_QuoteParser : public PabloKernel {
public:
    CSV_QuoteParser(LLVMTypeSystemInterface & ts, StreamSet * csvCCs, StreamSet * quotedData)
        : PabloKernel(ts, "CSV_QuoteParser",
                      {Binding{"csvCCs", csvCCs, FixedRate(), LookAhead(1)}},
                      {Binding{"quotedData", quotedData}}) {}
protected:
    void generatePabloMethod() override;
};

void CSV_QuoteParser::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvCCs");
    PabloAST * dquote = csvMarks[markDQ];
    PabloAST * dquote_odd = pb.createEveryNth(dquote, pb.getInteger(2));
    PabloAST * dquote_even = pb.createXor(dquote, dquote_odd);
    PabloAST * quote_escape = pb.createAnd(dquote_even, pb.createLookahead(dquote, 1));
    PabloAST * escaped_quote = pb.createAdvance(quote_escape, 1);
    PabloAST * start_dquote = pb.createXor(dquote_odd, escaped_quote);
    PabloAST * end_dquote = pb.createXor(dquote_even, quote_escape);
    PabloAST * start_data = pb.createAdvance(start_dquote, 1);
    PabloAST * quoted_data = pb.createIntrinsicCall(pablo::Intrinsic::SpanUpTo, {start_data, end_dquote});
    writeOutputStreamSet("quotedData",  std::vector<PabloAST*>{quoted_data, quote_escape});
}

class CSV_FieldParser : public PabloKernel {
public:
    CSV_FieldParser(LLVMTypeSystemInterface & ts, StreamSet * csvCCs, StreamSet * quotedData, 
                    StreamSet * recordSeparators, StreamSet * fieldStarts, StreamSet * fieldFollows)
        : PabloKernel(ts, "CSVparser",
                      {Binding{"csvCCs", csvCCs}, Binding{"quotedData", quotedData}},
                      {Binding{"recordSeparators", recordSeparators},
                       Binding{"fieldStarts", fieldStarts},
                       Binding{"fieldFollows", fieldFollows}}) {}
protected:
    void generatePabloMethod() override;
};

void CSV_FieldParser::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvCCs");
    std::vector<PabloAST *> quotedData = getInputStreamSet("quotedData");
    PabloAST * unquoted = pb.createNot(quotedData[0]);
    PabloAST * recordMarks = pb.createAnd(csvMarks[markLF], unquoted);
    PabloAST * fieldFollows = pb.createAnd(pb.createOr(csvMarks[markComma], csvMarks[markCR]), unquoted);
    // In the event that CSV records are terminated by bare LF and not CRLF, we update
    // fieldFollows to include this case.
    PabloAST * bareLF = pb.createAnd(pb.createNot(pb.createAdvance(csvMarks[markCR], 1)), recordMarks);
    fieldFollows = pb.createOr(fieldFollows, bareLF);
    PabloAST * fieldFinals = pb.createAnd(pb.createOr(csvMarks[markComma], csvMarks[markLF]), unquoted);
    PabloAST * fieldStarts = pb.createInFile(pb.createNot(pb.createAdvance(pb.createNot(fieldFinals), 1)));
    pb.createAssign(pb.createExtract(getOutputStreamVar("recordSeparators"), pb.getInteger(0)), recordMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("fieldStarts"), pb.getInteger(0)), fieldStarts);
    pb.createAssign(pb.createExtract(getOutputStreamVar("fieldFollows"), pb.getInteger(0)), fieldFollows);
}

CSV_Parser::CSV_Parser(PipelineBuilder & P, codepoint_t quoteCp, codepoint_t separatorCp) :
    mPB(P), mQuoteCp(quoteCp), mSeparatorCp(separatorCp), mCsvCCs(nullptr), mQuotedData(nullptr),
    mFieldStarts(nullptr), mFieldFollows(nullptr), mLineEnds(nullptr) {

    }

void CSV_Parser::setSource(StreamSet * basis) {
    mQuotedData = nullptr;
    mLineEnds = nullptr;
    mFieldStarts = nullptr;
    mFieldFollows = nullptr;
    std::vector<re::CC *> csv_marks(4);
    csv_marks[markLF] = re::makeCC(charLF);
    csv_marks[markCR] = re::makeCC(charCR);
    csv_marks[markDQ] = re::makeCC(mQuoteCp);
    csv_marks[markComma] = re::makeCC(mSeparatorCp);
    mCsvCCs = mPB.CreateStreamSet(4);
    mPB.CreateKernelCall<CharClassesKernel>(csv_marks, basis, mCsvCCs);
}

StreamSet * CSV_Parser::getCsvCCs() {return mCsvCCs;}
StreamSet * CSV_Parser::getQuotedData() {
    if (mQuotedData == nullptr) {
        mQuotedData = mPB.CreateStreamSet(2);
        mPB.CreateKernelCall<CSV_QuoteParser>(mCsvCCs, mQuotedData);
    }
    return mQuotedData;
}
StreamSet * CSV_Parser::getLineEnds() {
    if (mLineEnds == nullptr) {
        mFieldStarts = mPB.CreateStreamSet(1);
        mFieldFollows = mPB.CreateStreamSet(1);
        mLineEnds = mPB.CreateStreamSet(1);
        mPB.CreateKernelCall<CSV_FieldParser>(mCsvCCs, getQuotedData(), mLineEnds, mFieldStarts, mFieldFollows);
    }
    return mLineEnds;
}

StreamSet * CSV_Parser::getFieldStarts() {
    if (mFieldStarts == nullptr) {
        mFieldStarts = mPB.CreateStreamSet(1);
        mFieldFollows = mPB.CreateStreamSet(1);
        mLineEnds = mPB.CreateStreamSet(1);
        mPB.CreateKernelCall<CSV_FieldParser>(mCsvCCs, getQuotedData(), mLineEnds, mFieldStarts, mFieldFollows);
    }
    return mFieldStarts;
}
StreamSet * CSV_Parser::getFieldFollows() {
    if (mFieldFollows == nullptr) {
        mFieldStarts = mPB.CreateStreamSet(1);
        mFieldFollows = mPB.CreateStreamSet(1);
        mLineEnds = mPB.CreateStreamSet(1);
        mPB.CreateKernelCall<CSV_FieldParser>(mCsvCCs, getQuotedData(), mLineEnds, mFieldStarts, mFieldFollows);
    }
    return mFieldFollows;
}


class CSVparser : public PabloKernel {
public:
    CSVparser(LLVMTypeSystemInterface & ts, StreamSet * csvCCs, StreamSet * recordSeparators, StreamSet * fieldStarts, StreamSet * fieldFollows, StreamSet * quoteEscape)
        : PabloKernel(ts, "CSVparser",
                      {Binding{"csvCCs", csvCCs, FixedRate(), LookAhead(1)}},
                      {Binding{"recordSeparators", recordSeparators},
                       Binding{"fieldStarts", fieldStarts},
                       Binding{"fieldFollows", fieldFollows},
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
    PabloAST * fieldFollows = pb.createAnd(pb.createOr(csvMarks[markComma], csvMarks[markCR]), unquoted);
    // In the event that CSV records are terminated by bare LF and not CRLF, we update
    // fieldFollows to include this case.
    PabloAST * bareLF = pb.createAnd(pb.createNot(pb.createAdvance(csvMarks[markCR], 1)), recordMarks);
    fieldFollows = pb.createOr(fieldFollows, bareLF);
    PabloAST * fieldFinals = pb.createAnd(pb.createOr(csvMarks[markComma], csvMarks[markLF]), unquoted);
    PabloAST * fieldStarts = pb.createInFile(pb.createNot(pb.createAdvance(pb.createNot(fieldFinals), 1)));
    pb.createAssign(pb.createExtract(getOutputStreamVar("recordSeparators"), pb.getInteger(0)), recordMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("fieldStarts"), pb.getInteger(0)), fieldStarts);
    pb.createAssign(pb.createExtract(getOutputStreamVar("fieldFollows"), pb.getInteger(0)), fieldFollows);
    pb.createAssign(pb.createExtract(getOutputStreamVar("quoteEscape"), pb.getInteger(0)), quote_escape);
}

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)

void ParseCSV(PipelineBuilder & P, StreamSet * csvCCs,
              StreamSet * recordSeparators, StreamSet * fieldStarts, StreamSet * fieldFollows, StreamSet * quoteEscape) {

    P.CreateKernelCall<CSVparser>(csvCCs, recordSeparators, fieldStarts, fieldFollows, quoteEscape);
    SHOW_STREAM(recordSeparators);
    SHOW_STREAM(fieldStarts);
    SHOW_STREAM(fieldFollows);
    SHOW_STREAM(quoteEscape);
}

class SelectMultiField : public PabloKernel {
public:
    SelectMultiField(LLVMTypeSystemInterface & ts,
                     StreamSet * Record_separators,
                     StreamSet * fieldStarts,
                     StreamSet * fieldFollows,
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
                                   StreamSet * fieldStarts,
                                   StreamSet * fieldFollows,
                                   StreamSet * toKeep,
                                   const std::vector<unsigned> & columnNos,
                                   bool forCut)
: PabloKernel(ts, "SelectMultiField_" + colNoString(columnNos, forCut),
  {Binding{"Record_separators", Record_separators},
   Binding{"fieldStarts", fieldStarts},
   Binding{"fieldFollows", fieldFollows}},
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
    PabloAST * fieldStarts = pb.createExtract(getInputStreamVar("fieldStarts"), pb.getInteger(0));
    PabloAST * fieldFollows = pb.createExtract(getInputStreamVar("fieldFollows"), pb.getInteger(0));
    PabloAST * recordStarts = pb.createNot(pb.createAdvance(pb.createNot(Record_separators), 1));
    PabloAST * columnMark = recordStarts;
    PabloAST * toKeep = pb.createZeroes();
    unsigned nextCol = 0;
    for (unsigned i = 0; i < mColumnNos.size(); i++) {
        if (mColumnNos[i] > nextCol) {
            columnMark = pb.createIndexedAdvance(columnMark, fieldStarts, mColumnNos[i] - nextCol);
        }
        PabloAST * columnFollow = pb.createScanTo(columnMark, fieldFollows);
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

void ColumnSelectionMask(PipelineBuilder & P, StreamSet * Record_separators, StreamSet * fieldStarts, StreamSet * fieldFollows,
                         StreamSet * toKeep, const std::vector<unsigned> & columnNos, bool forCut) {
    P.CreateKernelCall<SelectMultiField>(Record_separators, fieldStarts, fieldFollows, toKeep, columnNos, forCut);
}

class EmptyFieldsKernel : public PabloKernel {
public:
    EmptyFieldsKernel(LLVMTypeSystemInterface & ts, StreamSet * csvCCs, StreamSet * fieldFollows,
                      StreamSet * EmptyFields);
protected:
    void generatePabloMethod() override;
};

EmptyFieldsKernel::EmptyFieldsKernel(LLVMTypeSystemInterface & ts, StreamSet * csvCCs, StreamSet * fieldFollows,
                                     StreamSet * EmptyMarks)
: PabloKernel(ts, "EmptyFieldsKernel" + csvCCs->shapeString(),
  {Binding{"csvCCs", csvCCs},
   Binding{"fieldFollows", fieldFollows, FixedRate(), LookAhead(1)}},
  {Binding{"EmptyMarks", EmptyMarks}})  {
}

void EmptyFieldsKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvCCs");
    PabloAST * fieldFollows = pb.createExtract(getInputStreamVar("fieldFollows"), pb.getInteger(0));
    PabloAST * fsNext = pb.createAdvance(fieldFollows, 1);
    PabloAST * quotedStart = pb.createAnd(fsNext, csvMarks[csv::markDQ]);
    PabloAST * quoted2 = pb.createAnd(pb.createAdvance(quotedStart, 1), csvMarks[csv::markDQ]);
    PabloAST * empties = pb.createAnd(pb.createOr(fsNext, pb.createAdvance(quoted2, 1)), fieldFollows);
    pb.createAssign(pb.createExtract(getOutputStreamVar("EmptyMarks"), pb.getInteger(0)), empties);
}

void GetEmptyFields(PipelineBuilder & P, StreamSet * csvCCs, StreamSet * fieldFollows,
                    StreamSet * EmptyFieldMarks) {
    P.CreateKernelCall<EmptyFieldsKernel>(csvCCs, fieldFollows, EmptyFieldMarks);
}
}
