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
#include <unicode/core/unicode_set.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>

using PipelineBuilder = kernel::PipelineBuilder;
using StreamSet = kernel::StreamSet;
using codepoint_t = UCD::codepoint_t;

namespace csv {

// Produce a set of 4 character class streams for CSV processing.
// LF - line feed stream
// CR - carriage return stream
// DQ - quote stream possibly overriden by csv::QuoteChar.
// Comma - comma stream, possibly overriden by csv::FieldDelimiter.
enum {markLF = 0, markCR = 1, markDQ = 2, markComma = 3};
void CSV_Lexer(PipelineBuilder & P, StreamSet * source, StreamSet * csvCCs);

// Parse a CSV file determining the record separators, field separators as
// well as any escaped quote marks.
void ParseCSV(PipelineBuilder & P, StreamSet * csvCCs, 
              StreamSet * recordSeparators, StreamSet * fieldSeparators, StreamSet * quoteEscape);

void ColumnSelectionMask(PipelineBuilder & P, StreamSet * Record_separators, StreamSet * Field_separators,
                         StreamSet * toKeep, const std::vector<unsigned> & columnNos);
}