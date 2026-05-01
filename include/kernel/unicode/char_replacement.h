/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <vector>
#include <kernel/pipeline/pipeline_builder.h>
#include <unicode/data/PropertyAliases.h>
#include <unicode/utf/transchar.h>

//
//  Given a Unicode character stream represented by a
//  full set of (21) parallel basis bit streams, apply the
//  named string override property from the Unicode
//  property data base to transform all relevant
//  characters of the stream, produced an updated
//  (expanded) character stream in parallel bit stream form.

kernel::StreamSet * U21_StringOverridePipeline(
    kernel::PipelineBuilder & P,
    UCD::property_t stringOverrideProperty,
    kernel::StreamSet * U21_basis);


//
//  Given a Unicode character stream represented by a
//  full set of (21) parallel basis bit streams, and a 
//  sets of bit translations which describe how characters
//  can be replaced by short strings, transform all relevant
//  characters of the stream, produced an updated
//  (expanded) character stream in parallel bit stream form.
//  The insert_length_bixnum is a set of CCs which specify
//  the extra spaces to be inserted for any characters which
//  map to multiposition replacements.    The translation sets
//  char_xlat_bitsets_by_position represent:
//  (a) CCs that specify how each bit of a source character changes
//      to produce the first character of the replacement string, or
//  (b) CCs that represent the bit value of the second or subsequent
//      character of the replacement string.

kernel::StreamSet *  U21_CharToShortStringPipeline(
    kernel::PipelineBuilder & P,
    unicode::BitTranslationSets & insert_length_bixnum,
    std::vector<unicode::BitTranslationSets> & char_xlat_bitsets_by_position,
    kernel::StreamSet * U21_basis);

