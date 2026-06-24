/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 *
 */
#include <ucd/utf/transchar.h>
#include <ucd/utf/utf_encoder.h>

using namespace UCD;

namespace unicode {

BitTranslationSets ComputeBitTranslationSets(const TranslationMap & map,
                                             XlateMode mode) {
    BitTranslationSets bit_xform_sets;
    // Otherwise compute and return.
    //
    // Basis set i is the set of all codepoints whose numerical enumeration code e
    // has bit i set, i.e., (e >> i) & 1 == 1.
    for (auto & p : map) {
        codepoint_t bit_diff =
            (mode == XlateMode::XorBit) ? p.second ^ p.first : p.second;
        unsigned bit = 0;
        while (bit_diff > 0) {
            if ((bit_diff & 1UL) == 1UL) {
                while (bit_xform_sets.size() <= bit) {
                    bit_xform_sets.push_back(UnicodeSet());
                }
                bit_xform_sets[bit].insert(p.first);
            }
            bit_diff >>= 1;
            bit++;
        }
    }
    return bit_xform_sets;
}

BitTranslationSets ComputeUTF8_insertionBixNum(const TranslationMap & map) {
    BitTranslationSets u8_insertion_bixnum;
    UTF_Encoder encoder;
    for (auto & p : map) {
        unsigned l1 = encoder.encoded_length(p.first);
        unsigned l2 = encoder.encoded_length(p.second);
        if (l2 > l1) {
            unsigned bit = 0;
            unsigned diff = l2 - l1;
            while (diff > 0) {
                if ((diff & 1UL) == 1UL) {
                    while (u8_insertion_bixnum.size() <= bit) {
                        u8_insertion_bixnum.push_back(UnicodeSet());
                    }
                    u8_insertion_bixnum[bit].insert(p.first);
                }
                diff >>= 1;
                bit++;
            }
        }
    }
    return u8_insertion_bixnum;
}
BitTranslationSets ComputeUTF8_deletionBixNum(const TranslationMap & map) {
    BitTranslationSets u8_deletion_bixnum;
    UTF_Encoder encoder;
    for (auto & p : map) {
        unsigned l1 = encoder.encoded_length(p.first);
        unsigned l2 = encoder.encoded_length(p.second);
        if (l1 > l2) {
            unsigned bit = 0;
            unsigned diff = l1 - l2;
            while (diff > 0) {
                if ((diff & 1UL) == 1UL) {
                    while (u8_deletion_bixnum.size() <= bit) {
                        u8_deletion_bixnum.push_back(UnicodeSet());
                    }
                    u8_deletion_bixnum[bit].insert(p.first);
                }
                diff >>= 1;
                bit++;
            }
        }
    }
    return u8_deletion_bixnum;
}
}
