/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once
#include <unicode/core/unicode_set.h>
#include <string>

using codepoint_t = UCD::codepoint_t;

class UTF_Encoder {
public:
    UTF_Encoder(unsigned bits = 8);
    unsigned encoded_length(codepoint_t cp);
    unsigned encoded_length(std::u32string s);
    codepoint_t max_codepoint_of_length(unsigned lgth);
    bool isLowCodePointAfterNthCodeUnit(codepoint_t cp, unsigned n);
    bool isHighCodePointAfterNthCodeUnit(codepoint_t cp, unsigned n);
    unsigned common_code_units(codepoint_t cp1, codepoint_t cp2);
    codepoint_t minCodePointWithCommonCodeUnits(codepoint_t cp, unsigned common);
    codepoint_t maxCodePointWithCommonCodeUnits(codepoint_t cp, unsigned common);
    codepoint_t minCodePointWithPrefix(unsigned code_unit);
    codepoint_t maxCodePointWithPrefix(unsigned code_unit);
    unsigned nthCodeUnit(codepoint_t cp, unsigned n);
    void setCodeUnitBits(unsigned bits) {mCodeUnitBits = bits;}
    unsigned getCodeUnitBits() {return mCodeUnitBits;}

private:
    unsigned mCodeUnitBits;
};
