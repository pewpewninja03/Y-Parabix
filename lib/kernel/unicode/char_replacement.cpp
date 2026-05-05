/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/unicode/char_replacement.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/unicode/charclasses.h>
#include <kernel/bitwise/bixlogic.h>
#include <ucd/data/PropertyObjects.h>
#include <ucd/data/PropertyObjectTable.h>
#include <re/adt/re_cc.h>

using namespace kernel;
using namespace llvm;

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

StreamSet * U21_StringOverridePipeline(PipelineBuilder & P,
                                       UCD::property_t string_map_property,
                                       StreamSet * U21_basis) {
    std::vector<unicode::BitTranslationSets> xfrms;
    unicode::BitTranslationSets insertion_bixnum;
    UCD::PropertyObject * propObj = UCD::getPropertyObject(string_map_property);
    if (UCD::CodePointPropertyObject * p = dyn_cast<UCD::CodePointPropertyObject>(propObj)) {
        xfrms.resize(1);
        xfrms[0] = p->GetBitTransformSets();
    } else if (UCD::StringOverridePropertyObject * p = dyn_cast<UCD::StringOverridePropertyObject>(propObj)) {
        for (unsigned i = 0; i < p->MaxUnicodeInsertLength(); i++) {
            xfrms.push_back(p->GetBitTransformSets(i));
        }
        insertion_bixnum = p->GetUnicodeInsertLengthBixNumSets();
    } else {
        llvm::report_fatal_error("Specified property is neither codepoint nor string override property.");
    }
    return U21_CharToShortStringPipeline(P, insertion_bixnum, xfrms, U21_basis);
}

StreamSet * U21_CharMapPipeline(PipelineBuilder & P,
    std::map<UCD::codepoint_t, std::u32string> replacementMap,
    StreamSet * U21_basis) {
    unicode::BitTranslationSets insertion_bixnum;
    unsigned maxLgth = 0;
    for (auto & mapping : replacementMap) {
        UCD::codepoint_t cp = mapping.first;
        if (mapping.second.size() > maxLgth) {
            maxLgth = mapping.second.size();
        }
        unsigned diff = mapping.second.size() - 1;
        unsigned bit = 0;
        while (diff > 0) {
            if ((diff & 1UL) == 1UL) {
                while (insertion_bixnum.size() <= bit) {
                    insertion_bixnum.push_back(UCD::UnicodeSet());
                }
                insertion_bixnum[bit].insert(cp);
            }
            diff >>= 1;
            bit++;
        }
    }
    std::vector<unicode::TranslationMap> xmaps(maxLgth);
    for (auto & mapping : replacementMap) {
        UCD::codepoint_t cp = mapping.first;
        std::u32string s = mapping.second;
        for (unsigned j = 0; j < s.size(); j++) {
            xmaps[j].emplace(cp, static_cast<UCD::codepoint_t>(s[j]));
        };
    }
    std::vector<unicode::BitTranslationSets> xfrms(maxLgth);
    xfrms[0] = unicode::ComputeBitTranslationSets(xmaps[0], unicode::XlateMode::XorBit);
    for (unsigned j = 1; j < maxLgth; j++) {
        xfrms[j] = unicode::ComputeBitTranslationSets(xmaps[j], unicode::XlateMode::LiteralBit);
    }
    return U21_CharToShortStringPipeline(P, insertion_bixnum, xfrms, U21_basis);
}

StreamSet * U21_CharToShortStringPipeline(PipelineBuilder & P,
    unicode::BitTranslationSets & insert_length_bixnum,
    std::vector<unicode::BitTranslationSets> & char_xlat_bitsets_by_position,
    StreamSet * U21_basis) {

    unsigned bix_bits = insert_length_bixnum.size();

    StreamSet * SpreadMask = nullptr;

    StreamSet * U21 = U21_basis;
    if (bix_bits > 0) {
        std::vector<re::CC *> insertion_ccs;
        for (auto & b : insert_length_bixnum) {
            insertion_ccs.push_back(re::makeCC(b, &cc::Unicode));
        }
        StreamSet * InsertBixNum = P.CreateStreamSet(bix_bits);
        P.CreateKernelCall<CharClassesKernel>(insertion_ccs, U21, InsertBixNum);
        SHOW_BIXNUM(InsertBixNum);

        SpreadMask = P.CreateStreamSet(1);
        InsertionSpreadMask(P, InsertBixNum, SpreadMask, kernel::InsertPosition::After);
        SHOW_STREAM(SpreadMask);

        StreamSet * ExpandedBasis = P.CreateStreamSet(21, 1);
        SpreadByMask(P, SpreadMask, U21, ExpandedBasis);
        SHOW_BIXNUM(ExpandedBasis);
        U21 = ExpandedBasis;
    }

    StreamSet * ResultBasis = U21;

    for (unsigned i = 0; i < char_xlat_bitsets_by_position.size(); i++) {
        std::vector<re::CC *> xfrm_ccs;
        for (auto & b : char_xlat_bitsets_by_position[i]) {
            xfrm_ccs.push_back(re::makeCC(b, &cc::Unicode));
        }
        StreamSet * BitXfmrs = P.CreateStreamSet(xfrm_ccs.size());
        P.CreateKernelCall<CharClassesKernel>(xfrm_ccs, U21, BitXfmrs);
        SHOW_BIXNUM(BitXfmrs);

        StreamSet * XfrmdBasis = P.CreateStreamSet(21, 1);

        if (i == 0) {
            XorCombine(P, ResultBasis, BitXfmrs, XfrmdBasis);
        } else {
            StreamSet * ForwardXfrmrs = P.CreateStreamSet(xfrm_ccs.size());
            P.CreateKernelCall<ShiftForward>(BitXfmrs, ForwardXfrmrs, i);
            SHOW_BIXNUM(ForwardXfrmrs);

            OrCombine(P, ResultBasis, ForwardXfrmrs, XfrmdBasis);
        }
        SHOW_BIXNUM(XfrmdBasis);

        ResultBasis = XfrmdBasis;
    }
    return ResultBasis;
}

