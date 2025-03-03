/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <unicode/utf/utf_encoder.h>
#include <llvm/Support/raw_ostream.h>

using codepoint_t = UCD::codepoint_t;

UTF_Encoder::UTF_Encoder(unsigned bits) : mCodeUnitBits(bits) {}

unsigned UTF_Encoder::encoded_length(codepoint_t cp) {
    if (mCodeUnitBits == 8) {
        if (cp <= 0x7F) return 1;
        else if (cp <= 0x7FF) return 2;
        else if (cp <= 0xFFFF) return 3;
        else return 4;
    } else if (mCodeUnitBits == 16) {
        if (cp <= 0xFFFF) return 1;
        else return 2;
    } else return 1;
}

unsigned UTF_Encoder::encoded_length(std::u32string s) {
    unsigned slgth = 0;
    for (size_t i = 0; i < s.size(); i++) {
        slgth += encoded_length(s[i]);
    }
    return slgth;
}

codepoint_t UTF_Encoder::max_codepoint_of_length(unsigned length) {
    if (mCodeUnitBits == 8) {
        if (length == 1) return 0x7F;
        else if (length == 2) return 0x7FF;
        else if (length == 3) return 0xFFFF;
        else {
            assert(length == 4);
            return 0x10FFFF;
        }
    } else if (mCodeUnitBits == 16) {
        if (length == 1) return 0xFFFF;
        else {
            assert(length == 2);
            return 0x10FFFF;
        }
    } else return 0x10FFFF;
}

bool UTF_Encoder::isLowCodePointAfterNthCodeUnit(codepoint_t cp, unsigned n) {
    if (cp == 0) return true;
    else if (encoded_length(cp - 1) < encoded_length(cp)) return true;
    else return nthCodeUnit(cp - 1, n) != nthCodeUnit(cp, n);
}

bool UTF_Encoder::isHighCodePointAfterNthCodeUnit(codepoint_t cp, unsigned n) {
    if (cp == 0x10FFFF) return true;
    else if (encoded_length(cp + 1) > encoded_length(cp)) return true;
    else return nthCodeUnit(cp + 1, n) != nthCodeUnit(cp, n);
}

unsigned UTF_Encoder::nthCodeUnit(codepoint_t cp, unsigned n) {
    const auto length = encoded_length(cp);
    if (mCodeUnitBits == 8) {
        if (n == 1) {
            switch (length) {
                case 1: return static_cast<unsigned>(cp);
                case 2: return static_cast<unsigned>(0xC0 | (cp >> 6));
                case 3: return static_cast<unsigned>(0xE0 | (cp >> 12));
                case 4: return static_cast<unsigned>(0xF0 | (cp >> 18));
            }
        }
        return static_cast<unsigned>(0x80 | ((cp >> (6 * (length - n))) & 0x3F));
    } else if (mCodeUnitBits == 16) {
        if (length == 1) {
            assert(n == 1);
            return static_cast<unsigned>(cp);
        }
        else if (n == 1)
            return static_cast<unsigned>(0xD800 | ((cp - 0x10000) >> 10));
        else
            return static_cast<unsigned>(0xDC00 | ((cp - 0x10000) & 0x3FF));
    } else {
        assert(n == 1);
        return static_cast<unsigned>(cp);
    }
}

codepoint_t UTF_Encoder::minCodePointWithCommonCodeUnits(codepoint_t cp, unsigned common) {
    const auto length = UTF_Encoder::encoded_length(cp);
    if (length == 1) return 0;
    if (mCodeUnitBits == 8) {
        const auto mask = (static_cast<codepoint_t>(1) << (length - common) * 6) - 1;
        const auto lo_cp = cp &~ mask;
        //
        return std::max(lo_cp, max_codepoint_of_length(length - 1) + 1);
    } else if (mCodeUnitBits == 16) {
        if (length == 1) return 0;
        else if (common == 1) {
            return cp & 0x1FFC00;
        }
        else return cp;
    } else return 0;
}

codepoint_t UTF_Encoder::maxCodePointWithCommonCodeUnits(codepoint_t cp, unsigned common) {
    const auto length = UTF_Encoder::encoded_length(cp);
    if (mCodeUnitBits == 8) {
        if (length == 1) return 0x7F;
        const auto mask = (static_cast<codepoint_t>(1) << (length - common) * 6) - 1;
        return std::min(cp | mask, max_codepoint_of_length(length));
    } else if (mCodeUnitBits == 16) {
        if (length == 1) return 0xFFFF;
        else if (common == 1) {
            return 0x10FFFF;
        }
        else return cp;
    } else return 0x10FFFF;
}

codepoint_t UTF_Encoder::minCodePointWithPrefix(unsigned prefix) {
    if (mCodeUnitBits == 8) {
        if (prefix <= 0xDFu) {
            return std::max((prefix & 0x1Fu) << 6, 0x80u);
        } else if (prefix <= 0xEFu) {
            return std::max((prefix & 0x0Fu) << 12, 0x800u);
        } else {
            return std::max((prefix & 0x07u) << 18, 0x10000u);
        }
    } else if (mCodeUnitBits == 16) {
        if ((prefix & 0xD8u) == 0xD8u) return 0x10000;
        return 0x0u;
    } else return 0u;
}

codepoint_t UTF_Encoder::maxCodePointWithPrefix(unsigned prefix) {
    if (mCodeUnitBits == 8) {
        if (prefix <= 0xDFu) {
            return minCodePointWithPrefix(prefix) | 0x3Fu;
        } else if (prefix <= 0xEFu) {
            return minCodePointWithPrefix(prefix) | 0xFFFu;
        } else {
            return std::min(minCodePointWithPrefix(prefix) | 0x3FFFFu, 0x10FFFFu);
        }
    } else if (mCodeUnitBits == 16) {
        if ((prefix & 0xD8u) == 0xD8u) return 0x10FFFFu;
        return 0xFFFFu;
    } else return 0x10FFFFu;
}


unsigned UTF_Encoder::common_code_units(codepoint_t cp1, codepoint_t cp2) {
    if (cp1 == cp2) return encoded_length(cp1);
    unsigned common = 0;
    while (nthCodeUnit(cp1, common+1) == nthCodeUnit(cp2, common+1)) {
        common++;
    }
    return common;
}
