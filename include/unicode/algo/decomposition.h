/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once
#include <string>
#include <unicode/core/UCD_Config.h>
#include <unicode/data/PropertyObjects.h>

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
protected:
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
};
}
