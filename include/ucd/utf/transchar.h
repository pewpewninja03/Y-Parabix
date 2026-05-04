/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <ucd/core/UCD_Config.h>
#include <unordered_map>
#include <ucd/core/unicode_set.h>

namespace unicode {

//
// Unicode Character Translation refers to operations such
// as simple case conversion, case folding, transliteration
// and other translations in which source Unicode characters
// are mapped to target Unicode characters.
//
using TranslationMap = std::unordered_map<UCD::codepoint_t, UCD::codepoint_t>;

//
// Unicode characters translations can be implemented in parallel
// using bit translation sets.   For each bit in the Unicode
// representation of source and target characters, the default bit
// translation set is the set of all codepoints such that the codepoint
// value at the given bit position differs from the target codepoint
// value at the same bit position.  Alternatively in LiteralBit
// mode, the bit translation set is based on the bits of the
// target codepoint only.
enum XlateMode {XorBit, LiteralBit};

//
// Unicode potentially requires 21 bit translation sets for each
// of the bit positions in Unicode.   However, if the higher
// order bits do not play a role in translation, they may be
// omitted.
//
using BitTranslationSets = std::vector<UCD::UnicodeSet>;
//
//  Given a TranslationMap return a vector of bit translations
//  for all bit positions up to the highest that participates
//  in the translation (i.e., the highest bit that differs
//  between the source and target codepoints in at least one
//  instance).
//
BitTranslationSets ComputeBitTranslationSets(const TranslationMap & map,
                                             XlateMode mode = XlateMode::XorBit);
//
//  Unicode character translation may be performed directly on
//  a UTF-8 basis bit representation, with the possible insertion
//  or deletion of up to 3 positions, i.e., insertion of 3 positions
//  for translations of a single-byte UTF-8 character to a 4-byte
//  UTF-8 sequence or vice versa.
//
//  Compute the UTF-8 insertion bixnum, which for each codepoint
//  cp is either 00, 01, 10, or 11 indicating that the trans[cp]
//  requires insertion of 0, 1, 2, or 3 bytes to the UTF-8 representation.
//  (In the case of 0 inserted bytes, deletion may be required).
//  General translations require a 2-bit bixnum, but a single-bit
//  bixnum may result when no instance of insertion of more than
//  one position is required.    If a translation requires no
//  insertions at all, an empty vector is returned.
//
BitTranslationSets ComputeUTF8_insertionBixNum(const TranslationMap & map);
//
//  Compute the UTF-8 deletion bixnum, which for each codepoint
//  cp is either 00, 01, 10, or 11 indicating that the trans[cp]
//  requires deletion of 0, 1, 2, or 3 bytes to the UTF-8 representation.
//  (In the case of 0 deleted bytes, insertion may be required).
//  General translations require a 2-bit bixnum, but a single-bit
//  bixnum may result when no instance of deletion of more than
//  one position is required.    If a translation requires no
//  deletions at all, an empty vector is returned.
//
BitTranslationSets ComputeUTF8_deletionBixNum(const TranslationMap & map);
}
