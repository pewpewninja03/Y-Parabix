/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once
#include <string>
#include <vector>
#include <unicode/core/UCD_Config.h>
#include <unicode/core/unicode_set.h>
#include <unicode/utf/transchar.h>
#include <unicode/data/PropertyObjects.h>

//
// Constants for algorithmic transformation of Hangul graphemes between
// precomposed and decomposed forms.   See Section 3.12 of the Unicode standard.
//
namespace Hangul {
const UCD::codepoint_t SBase = 0xAC00;
const UCD::codepoint_t LBase = 0x1100;
const UCD::codepoint_t VBase = 0x1161;
const UCD::codepoint_t TBase = 0x11A7;
const unsigned LCount = 19;
const unsigned VCount = 21;
const unsigned TCount = 28;
const unsigned NCount = 588;
const unsigned SCount = 11172;
const UCD::codepoint_t MaxPrecomposed = SBase + SCount - 1;
}

namespace UCD {
//  Decomposition may use either canonical or compatibility mappings,
//  possibly combined with casefolding.
enum DecompositionOptions : int { NFD = 0, CaseFold = 1, NFKD = 2 };

// An NFD Engine can be used to repeatedly perform decomposition
// using selected decomposition options.
class NFD_Engine {
public:
    NFD_Engine(DecompositionOptions opt);
    std::u32string decompose(std::u32string);
    /* Helpers to convert and append an individual codepoint or a u32string
    to an existing NFD_string.   The process performs any necessary
    reordering of marks of the existing string and the appended data
    to ensure that the result is overall in NFD form.
    NFD_Engine(CaseFold).NFD_append1(s, cp);
    */
    void NFD_append1(std::u32string & NFD_string, codepoint_t cp);
    void NFD_append(std::u32string & NFD_string, std::u32string & to_convert);
    bool reordering_needed(std::u32string & prefix, codepoint_t suffix_cp);

    // Return the sets that can be used to produce an insert length bixnum
    // for the given NFD or NFKD transform.
    unicode::BitTranslationSets UnicodeInsertLengthBixNumSets();
    // Determine the bit transformations required to make characters
    // into their decompositions.
    std::vector<unicode::BitTranslationSets> UnicodeBitTransformSets();
protected:
    void prepareUnicodeBitTranslationData();
private:
    DecompositionOptions mOptions;
    EnumeratedPropertyObject * decompTypeObj;
    StringPropertyObject * decompMappingObj;
    EnumeratedPropertyObject * cccObj;
    StringOverridePropertyObject * caseFoldObj;
    const UnicodeSet canonicalMapped;
    const UnicodeSet cc0Set;
    const UnicodeSet selfNFKD;
    const UnicodeSet selfCaseFold;
    const UnicodeSet HangulPrecomposed;
    unicode::BitTranslationSets decomposition_insert_length_bixnum_sets;
    std::vector<unicode::BitTranslationSets> bitXlatSets;
};
}
