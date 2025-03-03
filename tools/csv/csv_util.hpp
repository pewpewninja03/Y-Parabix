#include <fstream>
#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pe_ones.h>
#include <pablo/pablo_toolchain.h>
#include <pablo/bixnum/bixnum.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/intrusive/detail/math.hpp>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>

using namespace pablo;
using namespace kernel;

using boost::intrusive::detail::ceil_log2;

std::vector<std::string> parse_CSV_headers(std::string headerString) {
    std::vector<std::string> headers;
    boost::algorithm::split(headers, headerString, [] (char c) {return (c == ',');});
    for (unsigned i = 0; i < headers.size(); i++) {
        boost::algorithm::trim(headers[i]);
    }
    return headers;
}

std::vector<std::string> get_CSV_headers(std::string filename) {
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

std::vector<std::string> JSONfieldPrefixes(std::vector<std::string> fieldNames) {
    std::vector<std::string> tmp;
    if (fieldNames.size() == 0) return tmp;
    for (unsigned i = 0; i < fieldNames.size(); i++) {
        tmp.push_back("\"" + fieldNames[i] + "\":\"");
    }
    tmp[0] = "{" + tmp[0];
    return tmp;
}

char charLF = 0xA;
char charCR = 0xD;
char charDQ = 0x22;
char charComma = 0x2C;

class CSVlexer : public PabloKernel {
public:
    CSVlexer(LLVMTypeSystemInterface & ts, StreamSet * Source, StreamSet * CSVlexical)
        : PabloKernel(ts, "CSVlexer",
                      {Binding{"Source", Source}},
                      {Binding{"CSVlexical", CSVlexical, FixedRate(), Add1()}}) {}
protected:
    void generatePabloMethod() override;
};

enum {markLF = 0, markCR = 1, markDQ = 2, markComma = 3, markEOF = 4};

void CSVlexer::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::unique_ptr<cc::CC_Compiler> ccc;
    ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(getInputStreamSet("Source"));
    PabloAST * LF = ccc->compileCC(re::makeCC(charLF, &cc::Byte), pb);
    PabloAST * CR = ccc->compileCC(re::makeCC(charCR, &cc::Byte), pb);
    PabloAST * DQ = ccc->compileCC(re::makeCC(charDQ, &cc::Byte), pb);
    PabloAST * Comma = ccc->compileCC(re::makeCC(charComma, &cc::Byte), pb);
    PabloAST * EOFbit = pb.createAtEOF(pb.createAdvance(pb.createOnes(), 1));
    Var * lexOut = getOutputStreamVar("CSVlexical");
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(markLF)), LF);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(markCR)), CR);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(markDQ)), DQ);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(markComma)), Comma);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(markEOF)), EOFbit);
}

class CSVparser : public PabloKernel {
public:
    CSVparser(LLVMTypeSystemInterface & ts, StreamSet * csvMarks, StreamSet * recordSeparators, StreamSet * fieldSeparators, StreamSet * quoteEscape)
        : PabloKernel(ts, "CSVparser",
                      {Binding{"csvMarks", csvMarks, FixedRate(), LookAhead(1)}},
                      {Binding{"recordSeparators", recordSeparators},
                       Binding{"fieldSeparators", fieldSeparators},
                       Binding{"quoteEscape", quoteEscape}}) {}
protected:
    void generatePabloMethod() override;
};

void CSVparser::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvMarks");
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

class CSVdataFieldMask : public PabloKernel {
public:
    CSVdataFieldMask(LLVMTypeSystemInterface & ts, StreamSet * csvMarks, StreamSet * recordSeparators, StreamSet * quoteEscape, StreamSet * toKeep, bool deleteHeader = true)
        : PabloKernel(ts, "CSVdataFieldMask" + std::to_string(deleteHeader),
                      {Binding{"csvMarks", csvMarks, FixedRate(), LookAhead(1)},
                       Binding{"recordSeparators", recordSeparators},
                       Binding{"quoteEscape", quoteEscape}},
                      {Binding{"toKeep", toKeep}})
    , mDeleteHeader(deleteHeader) {}
protected:
    void generatePabloMethod() override;
    bool mDeleteHeader;
};

void CSVdataFieldMask::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvMarks");
    PabloAST * recordMarks = pb.createExtract(getInputStreamVar("recordSeparators"), pb.getInteger(0));
    PabloAST * quoteEscape = pb.createExtract(getInputStreamVar("quoteEscape"), pb.getInteger(0));
    PabloAST * CRbeforeLF = pb.createAnd(csvMarks[markCR], pb.createLookahead(csvMarks[markLF], 1));
    PabloAST * escaped_quote = pb.createAdvance(quoteEscape, 1);
    PabloAST * fieldQuotes = pb.createAnd(csvMarks[markDQ], pb.createNot(pb.createOr(quoteEscape, escaped_quote)));
    PabloAST * toDelete = pb.createOr(CRbeforeLF, fieldQuotes);
    if (mDeleteHeader) {
        PabloAST * afterHeader = pb.createMatchStar(pb.createAdvance(recordMarks, 1), pb.createOnes());
        toDelete = pb.createOr(toDelete, pb.createNot(afterHeader));
    }
    // Delete the final LF position, so that we won't generate a template string at EOF.
    toDelete = pb.createOr(toDelete, pb.createAnd(recordMarks, pb.createLookahead(csvMarks[markEOF], 1)));
    // Also delete the final EOFbit position generated by the Add1 attribute of the CSV lexer to avoid a null.
    toDelete = pb.createOr(toDelete, csvMarks[markEOF]);
    PabloAST * toKeep = pb.createInFile(pb.createNot(toDelete));
    pb.createAssign(pb.createExtract(getOutputStreamVar("toKeep"), pb.getInteger(0)), toKeep);
}

//
//  FieldNumberingKernel(N) 
//  two input streams: record marks, field marks, N fields per record
//  output: at the start position after each mark, a bixnum value equal to the
//          sequential field number (counting from 1 at each record start).
//

class FieldNumberingKernel : public PabloKernel {
public:
    FieldNumberingKernel(LLVMTypeSystemInterface & ts, StreamSet * SeparatorNum, StreamSet * RecordMarks, StreamSet * FieldBixNum);
protected:
    void generatePabloMethod() override;
    unsigned mNumberingBits;
};

FieldNumberingKernel::FieldNumberingKernel(LLVMTypeSystemInterface & ts, StreamSet * SeparatorNum, StreamSet * RecordMarks, StreamSet * FieldBixNum)
   : PabloKernel(ts, "FieldNumbering" + std::to_string(SeparatorNum->getNumElements()),
                 {Binding{"RecordMarks", RecordMarks}, Binding{"SeparatorNum", SeparatorNum}}, {Binding{"FieldBixNum", FieldBixNum}}),
   mNumberingBits(SeparatorNum->getNumElements()) { }

void FieldNumberingKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * recordMarks = getInputStreamSet("RecordMarks")[0];   //  1 at record positions, 0 elsewhere
    BixNum separatorNum = getInputStreamSet("SeparatorNum"); //  consecutively numbered from 0
    BixNum increment(2);
    increment[0] = recordMarks;   //  Add 1 at record positions
    increment[1] = pb.createNot(recordMarks);  // Add 2 at field mark positions.
    BixNum fieldNumbering = bnc.AddFull(separatorNum, increment);
    writeOutputStreamSet("FieldBixNum", fieldNumbering);
}

class CSV_Char_Replacement : public PabloKernel {
public:
    CSV_Char_Replacement(LLVMTypeSystemInterface & ts, StreamSet * quoteEscape, StreamSet * basis,
                         StreamSet * translatedBasis)
        : PabloKernel(ts, "CSV_Char_Replacement",
                      {Binding{"quoteEscape", quoteEscape}, Binding{"basis", basis}},
                      {Binding{"translatedBasis", translatedBasis}}) {}
protected:
    void generatePabloMethod() override;
};

void CSV_Char_Replacement::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * quoteEscape = getInputStreamSet("quoteEscape")[0];
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    //
    // Translate "" to \"  ASCII value of " = 0x22, ASCII value of \ = 0x5C
    std::vector<PabloAST *> translated_basis(8, nullptr);
    translated_basis[0] = basis[0];
    translated_basis[1] = pb.createXor(basis[1], quoteEscape);
    translated_basis[2] = pb.createXor(basis[2], quoteEscape);
    translated_basis[3] = pb.createXor(basis[3], quoteEscape);
    translated_basis[4] = pb.createXor(basis[4], quoteEscape);
    translated_basis[5] = pb.createXor(basis[5], quoteEscape);
    translated_basis[6] = pb.createXor(basis[6], quoteEscape);
    translated_basis[7] = basis[7];
    writeOutputStreamSet("translatedBasis", translated_basis);
}

class AddFieldSuffix : public PabloKernel {
public:
    AddFieldSuffix(LLVMTypeSystemInterface & ts, StreamSet * suffixSpreadMask, StreamSet * basis,
                         StreamSet * updatedBasis)
        : PabloKernel(ts, "AddFieldSuffix",
                      {Binding{"suffixSpreadMask", suffixSpreadMask}, Binding{"basis", basis}},
                      {Binding{"updatedBasis", updatedBasis}}) {}
protected:
    void generatePabloMethod() override;
};

void AddFieldSuffix::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * mask = getInputStreamSet("suffixSpreadMask")[0];
    //
    // The suffixSpreadMask marks position for suffix data with
    // either a single 0 (insertion position before comma) or a
    // run of three 0s (insertion position before newline).
    // A special case is a 0 at start of file.
    PabloAST * mask1 = pb.createNot(mask);  // convert to 1-based mask
    PabloAST * quotePos = pb.createAnd(pb.createNot(pb.createAdvance(mask1, 1)), mask1);
    PabloAST * RbracePos = pb.createAnd(pb.createAdvance(quotePos, 1), mask1);
    PabloAST * afterRbrace = pb.createAdvance(RbracePos, 1);
    PabloAST * commaPos = pb.createAnd(afterRbrace, mask1);

    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    // Quotes are ASCII 0x22 - only need to set bits 5 and 2.
    basis[1] = pb.createOr(basis[1], quotePos);
    basis[5] = pb.createOr(basis[5], quotePos);
    //
    // Add } (ASCII 7d) after " at recordSeparator.
    basis[0] = pb.createOr(basis[0], RbracePos);
    basis[2] = pb.createOr(basis[2], RbracePos);
    basis[3] = pb.createOr(basis[3], RbracePos);
    basis[4] = pb.createOr(basis[4], RbracePos);
    basis[5] = pb.createOr(basis[5], RbracePos);
    basis[6] = pb.createOr(basis[6], RbracePos);
    //
    // Add , (ASCII 2c) after } at recordSeparator.
    basis[2] = pb.createOr(basis[2], commaPos);
    basis[3] = pb.createOr(basis[3], commaPos);
    basis[5] = pb.createOr(basis[5], commaPos);
    Var * translatedVar = getOutputStreamVar("updatedBasis");
    for (unsigned i = 0; i < 8; i++) {
        pb.createAssign(pb.createExtract(translatedVar, pb.getInteger(i)), basis[i]);
    }
}
