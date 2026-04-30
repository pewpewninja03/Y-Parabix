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
#include <re/adt/re_cc.h>

using namespace kernel;
using namespace llvm;

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

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

