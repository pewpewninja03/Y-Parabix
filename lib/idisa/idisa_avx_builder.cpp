/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_avx_builder.h>
#include <boost/intrusive/detail/math.hpp>
#include <toolchain/toolchain.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Intrinsics.h>
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(10, 0, 0)
#include <llvm/IR/IntrinsicsX86.h>
#endif
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(17, 0, 0)
#include <llvm/TargetParser/Host.h>
#elif LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(11, 0, 0)
#include <llvm/Support/Host.h>
#endif
using boost::intrusive::detail::floor_log2;

#define ADD_IF_FOUND(Flag, Value) if (features.lookup(Value)) featureSet.set((size_t)Feature::Flag)

using namespace llvm;

namespace IDISA {

std::string IDISA_AVX_Builder::getBuilderUniqueName() {
    return mBitBlockWidth != 256 ? "AVX_" + std::to_string(mBitBlockWidth) : "AVX";
}

Value * IDISA_AVX_Builder::hsimd_signmask(unsigned fw, Value * a) {
    // AVX2 special cases
    if (mBitBlockWidth == 256) {
        if (fw == 64) {
            Function * signmask_f64func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx_movmsk_pd_256);
            Type * bitBlock_f64type = FixedVectorType::get(getDoubleTy(), 256 / 64);
            Value * a_as_pd = CreateBitCast(a, bitBlock_f64type);
            return CreateCall(signmask_f64func->getFunctionType(), signmask_f64func, a_as_pd);
        } else if (fw == 32) {
            Function * signmask_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx_movmsk_ps_256);
            Type * bitBlock_f32type = FixedVectorType::get(getFloatTy(), 256 / 32);
            Value * a_as_ps = CreateBitCast(a, bitBlock_f32type);
            return CreateCall(signmask_f32func->getFunctionType(), signmask_f32func, a_as_ps);
        }
    } else if (mBitBlockWidth == 512) {
        if (fw == 64) {
            Type * bitBlock_f32type = FixedVectorType::get(getFloatTy(), mBitBlockWidth / 32);
            Value * a_as_ps = CreateBitCast(a, bitBlock_f32type);
            Constant * indicies[8];
            for (unsigned i = 0; i < 8; i++) {
                indicies[i] = getInt32(2 * i + 1);
            }
            Value * packh = CreateShuffleVector(a_as_ps, UndefValue::get(bitBlock_f32type), ConstantVector::get({indicies, 8}));
            Type * halfBlock_f32type = FixedVectorType::get(getFloatTy(), mBitBlockWidth/64);
            Value * pack_as_ps = CreateBitCast(packh, halfBlock_f32type);
            Function * signmask_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx_movmsk_ps_256);
            return CreateCall(signmask_f32func->getFunctionType(), signmask_f32func, pack_as_ps);
        }
    }
    // Otherwise use default SSE2 logic.
    return IDISA_SSE2_Builder::hsimd_signmask(fw, a);
}

Value * IDISA_AVX_Builder::CreateZeroHiBitsFrom(Value * bits, Value * pos, const Twine Name) {
    if (hasFeature(Feature::AVX_BMI)) {
        Type * const Ty = bits->getType();
        if (Ty == getInt64Ty()) {
            Function * bzhi_64 = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_bzhi_64);
            return CreateCall(bzhi_64->getFunctionType(), bzhi_64, {bits, pos}, Name);
        }
        if (Ty == getInt32Ty()) {
            Function * bzhi_32 = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_bzhi_32);
            return CreateCall(bzhi_32->getFunctionType(), bzhi_32, {bits, pos}, Name);
        }
    }
    return CBuilder::CreateZeroHiBitsFrom(bits, pos, Name);
}

Value * IDISA_AVX_Builder::CreatePextract(Value * bits, Value * mask, const Twine Name) {
    if (hasFeature(Feature::AVX_BMI2)) {
        Type * Ty = bits->getType();
        unsigned width = Ty->getPrimitiveSizeInBits();
        if (width == 64) {
            Function * pext64 = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pext_64);
            return CreateCall(pext64->getFunctionType(), pext64, {bits, mask}, Name);
        }
        if (width <= 32) {
            Function * pext32 = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pext_32);
            if (width < 32) {
                assert (bits->getType()->getIntegerBitWidth() <= 32);
                bits = CreateZExt(bits, getInt32Ty());
                assert (mask->getType()->getIntegerBitWidth() <= 32);
                mask = CreateZExt(mask, getInt32Ty());
            }
            Value * r = CreateCall(pext32->getFunctionType(), pext32, {bits, mask}, Name);
            if (width == 32) return r;
            return CreateTrunc(r, bits->getType());
        }
    }
    return IDISA_Builder::CreatePextract(mask, bits, Name);
}

Value * IDISA_AVX_Builder::CreatePdeposit(Value * bits, Value * mask, const Twine Name) {
    if (hasFeature(Feature::AVX_BMI2)) {
        Type * Ty = bits->getType();
        unsigned width = Ty->getPrimitiveSizeInBits();
        if (width == 64) {
            Function * pdep64 = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pdep_64);
            return CreateCall(pdep64->getFunctionType(), pdep64, {bits, mask}, Name);
        }
        if (width <= 32) {
            Function * pdep32 = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pdep_32);
            if (width < 32) {
                assert (bits->getType()->getIntegerBitWidth() <= 32);
                bits = CreateZExt(bits, getInt32Ty());
                assert (mask->getType()->getIntegerBitWidth() <= 32);
                mask = CreateZExt(mask, getInt32Ty());
            }
            Value * r = CreateCall(pdep32->getFunctionType(), pdep32, {bits, mask}, Name);
            if (width == 32) return r;
            return CreateTrunc(r, bits->getType());
        }
    }
    return IDISA_Builder::CreatePdeposit(mask, bits, Name);
}

std::string IDISA_AVX2_Builder::getBuilderUniqueName() {
    return mBitBlockWidth != 256 ? "AVX2_" + std::to_string(mBitBlockWidth) : "AVX2";
}

Value * IDISA_AVX2_Builder::hsimd_packh(unsigned fw, Value * a, Value * b) {
    if ((fw > 8) && (fw <= 64)) {
        Value * aVec = fwCast(fw / 2, a);
        Value * bVec = fwCast(fw / 2, b);
        const auto field_count = 2 * mBitBlockWidth / fw;
        SmallVector<Constant *, 32> Idxs(field_count);
        const auto H = (field_count / 2);
        const auto Q = (field_count / 4);
        for (unsigned i = 0; i < Q; i++) {
            Idxs[i] = getInt32(2 * i);
            Idxs[i + Q] = getInt32((2 * i) + 1);
            Idxs[i + H] = getInt32((2 * i) + H);
            Idxs[i + H + Q] = getInt32((2 * i) + 1 + H);
        }
        Constant * const IdxVec = ConstantVector::get(Idxs);
        Value * shufa = CreateShuffleVector(aVec, aVec, IdxVec);
        Value * shufb = CreateShuffleVector(bVec, bVec, IdxVec);
        return hsimd_packh(mBitBlockWidth / 2, shufa, shufb);
    }
    // Otherwise use default SSE logic.
    return IDISA_SSE2_Builder::hsimd_packh(fw, a, b);
}

Value * IDISA_AVX2_Builder::hsimd_packl(unsigned fw, Value * a, Value * b) {
    if ((fw > 8) && (fw <= 64)) {
        Value * aVec = fwCast(fw / 2, a);
        Value * bVec = fwCast(fw / 2, b);
        const auto field_count = 2 * mBitBlockWidth / fw;
        SmallVector<Constant *, 16> Idxs(field_count);
        const auto H = (field_count / 2);
        const auto Q = (field_count / 4);
        for (unsigned i = 0; i < Q; i++) {
            Idxs[i] = getInt32(2 * i);
            Idxs[i + Q] = getInt32((2 * i) + 1);
            Idxs[i + H] = getInt32((2 * i) + H);
            Idxs[i + H + Q] = getInt32((2 * i) + H + 1);
        }
        Constant * const IdxVec = ConstantVector::get(Idxs);
        Value * shufa = CreateShuffleVector(aVec, aVec, IdxVec);
        Value * shufb = CreateShuffleVector(bVec, bVec, IdxVec);
        return hsimd_packl(mBitBlockWidth / 2, shufa, shufb);
    }
    // Otherwise use default SSE logic.
    return IDISA_SSE2_Builder::hsimd_packl(fw, a, b);
}

Value * IDISA_AVX2_Builder::esimd_mergeh(unsigned fw, Value * a, Value * b) {
    if (getVectorBitWidth(a) == mNativeBitBlockWidth) {
        if ((fw == 1) || (fw == 2)) {
            // Bit interleave using shuffle.
            Function * shufFn = Intrinsic::getDeclaration(getModule(),  Intrinsic::x86_avx2_pshuf_b);
            // Make a shuffle table that translates the lower 4 bits of each byte in
            // order to spread out the bits: xxxxdcba => .d.c.b.a
            // We use two copies of the table for the AVX2 _mm256_shuffle_epi8
            Constant * interleave_table = bit_interleave_byteshuffle_table(fw);
            // Merge the bytes.
            Value * byte_merge = esimd_mergeh(8, a, b);
            Value * low_bits = CreateCall(shufFn->getFunctionType(), shufFn, {interleave_table,  fwCast(8, simd_select_lo(8, byte_merge))});
            Value * high_bits = simd_slli(16, CreateCall(shufFn->getFunctionType(), shufFn, {interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))}), fw);
            // For each 16-bit field, interleave the low bits of the two bytes.
            low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8-fw));
            // For each 16-bit field, interleave the high bits of the two bytes.
            high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8-fw));
            return simd_or(low_bits, high_bits);
        }
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(6, 0, 0)
        if (fw == 128) {
            Function * vperm2i128func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx2_vperm2i128);
            return CreateCall(vperm2i128func->getFunctionType(), vperm2i128func, {fwCast(64, a), fwCast(64, b), getInt8(0x31)});
        }
#endif
    }
    // Otherwise use default SSE logic.
    return IDISA_SSE2_Builder::esimd_mergeh(fw, a, b);
}

Value * IDISA_AVX2_Builder::esimd_mergel(unsigned fw, Value * a, Value * b) {
    if (getVectorBitWidth(a) == mNativeBitBlockWidth) {
        if ((fw == 1) || (fw == 2)) {
            // Bit interleave using shuffle.
            Function * shufFn = Intrinsic::getDeclaration(getModule(),  Intrinsic::x86_avx2_pshuf_b);
            // Make a shuffle table that translates the lower 4 bits of each byte in
            // order to spread out the bits: xxxxdcba => .d.c.b.a
            // We use two copies of the table for the AVX2 _mm256_shuffle_epi8
            Constant * interleave_table = bit_interleave_byteshuffle_table(fw);
            // Merge the bytes.
            Value * byte_merge = esimd_mergel(8, a, b);
            Value * low_bits = CreateCall(shufFn->getFunctionType(), shufFn, {interleave_table,  fwCast(8, simd_select_lo(8, byte_merge))});
            Value * high_bits = simd_slli(16, CreateCall(shufFn->getFunctionType(), shufFn, {interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))}), fw);
            // For each 16-bit field, interleave the low bits of the two bytes.
            low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8-fw));
            // For each 16-bit field, interleave the high bits of the two bytes.
            high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8-fw));
            return simd_or(low_bits, high_bits);
        }
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(6, 0, 0)
        if ((fw == 128) && (mBitBlockWidth == 256)) {
            Function * vperm2i128func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx2_vperm2i128);
            return CreateCall(vperm2i128func->getFunctionType(), vperm2i128func, {fwCast(64, a), fwCast(64, b), getInt8(0x20)});
        }
    #endif
    }
    // Otherwise use default SSE logic.
    return IDISA_SSE_Builder::esimd_mergel(fw, a, b);
}

Value * IDISA_AVX2_Builder::hsimd_packl_in_lanes(unsigned lanes, unsigned fw, Value * a, Value * b) {
    if ((fw == 16)  && (lanes == 2)) {
        Function * vpackuswbfunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx2_packuswb);
        Value * a_low = fwCast(16, simd_and(a, simd_lomask(fw)));
        Value * b_low = fwCast(16, simd_and(b, simd_lomask(fw)));
        return CreateCall(vpackuswbfunc->getFunctionType(), vpackuswbfunc, {a_low, b_low});
    }
    // Otherwise use default SSE logic.
    return IDISA_SSE_Builder::hsimd_packl_in_lanes(lanes, fw, a, b);
}

Value * IDISA_AVX2_Builder::hsimd_packh_in_lanes(unsigned lanes, unsigned fw, Value * a, Value * b) {
    if ((fw == 16)  && (lanes == 2)) {
        Function * vpackuswbfunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx2_packuswb);
        Value * a_low = simd_srli(fw, a, fw/2);
        Value * b_low = simd_srli(fw, b, fw/2);
        return CreateCall(vpackuswbfunc->getFunctionType(), vpackuswbfunc, {a_low, b_low});
    }
    // Otherwise use default SSE logic.
    return IDISA_SSE_Builder::hsimd_packh_in_lanes(lanes, fw, a, b);
}


Value * IDISA_AVX2_Builder::hsimd_packus(unsigned fw, Value * a, Value * b) {
    if (((fw == 32) || (fw == 16)) && (getVectorBitWidth(a) == AVX_width)) {
        Function * pack_func = Intrinsic::getDeclaration(getModule(), fw == 16 ? Intrinsic::x86_avx2_packuswb : Intrinsic::x86_avx2_packusdw);
        Value * packed = fwCast(64, CreateCall(pack_func->getFunctionType(), pack_func, {fwCast(fw, a), fwCast(fw, b)}));
        auto field_count = AVX_width/64;
        SmallVector<Constant *, 4> Idxs(field_count);
        for (unsigned int i = 0; i < field_count/2; i++) {
            Idxs[i] = getInt32(2*i);
            Idxs[i + field_count/2] = getInt32(2*i + 1);
        }
        Constant * shuffleMask = ConstantVector::get(Idxs);
        return CreateShuffleVector(packed, UndefValue::get(fwVectorType(64)), shuffleMask);
    }
    // Otherwise use default logic.
    return IDISA_Builder::hsimd_packus(fw, a, b);
}

Value * IDISA_AVX2_Builder::hsimd_packss(unsigned fw, Value * a, Value * b) {
    if (((fw == 32) || (fw == 16)) && (getVectorBitWidth(a) == AVX_width)) {
        Function * pack_func = Intrinsic::getDeclaration(getModule(), fw == 16 ? Intrinsic::x86_avx2_packsswb : Intrinsic::x86_avx2_packssdw);
        Value * packed = fwCast(64, CreateCall(pack_func->getFunctionType(), pack_func, {fwCast(fw, a), fwCast(fw, b)}));
        auto field_count = AVX_width/64;
        SmallVector<Constant *, 4> Idxs(field_count);
        for (unsigned int i = 0; i < field_count/2; i++) {
            Idxs[i] = getInt32(2*i);
            Idxs[i + field_count/2] = getInt32(2*i + 1);
        }
        Constant * shuffleMask = ConstantVector::get(Idxs);
        return CreateShuffleVector(packed, UndefValue::get(fwVectorType(64)), shuffleMask);
    }
    // Otherwise use default logic.
    return IDISA_Builder::hsimd_packus(fw, a, b);
}

std::pair<Value *, Value *> IDISA_AVX2_Builder::bitblock_add_with_carry(Value * e1, Value * e2, Value * carryin) {
    // using LONG_ADD
    Type * carryTy = carryin->getType();
    if (carryTy == mBitBlockType) {
        carryin = mvmd_extract(32, carryin, 0);
    } else if (carryTy->getIntegerBitWidth() < 32) {
        assert (carryTy->isIntegerTy());
        carryin = CreateZExt(carryin, getInt32Ty());
    }
    Value * carrygen = simd_and(e1, e2);
    Value * carryprop = simd_or(e1, e2);
    Value * digitsum = simd_add(64, e1, e2);
    Value * digitcarry = simd_or(carrygen, simd_and(carryprop, CreateNot(digitsum)));
    Value * carryMask = hsimd_signmask(64, digitcarry);
    Value * carryMask2 = CreateOr(CreateAdd(carryMask, carryMask), carryin);
    Value * bubble = simd_eq(64, digitsum, allOnes());
    Value * bubbleMask = hsimd_signmask(64, bubble);
    Value * incrementMask = CreateXor(CreateAdd(bubbleMask, carryMask2), bubbleMask);
    Value * increments = esimd_bitspread(64,incrementMask);
    Value * sum = simd_add(64, digitsum, increments);
    Value * carry_out = CreateLShr(incrementMask, mBitBlockWidth / 64);
    assert (carry_out->getType()->getIntegerBitWidth() == 32);
    if (carryTy == mBitBlockType) {
        carry_out = CreateZExtOrTrunc(carry_out, mBitBlockType->getElementType());
        carry_out = CreateInsertElement(ConstantVector::getNullValue(mBitBlockType), carry_out, getInt32(0));
    } else if (carryTy != carry_out->getType()) {
        carry_out = CreateZExtOrTrunc(carry_out, carryTy);
    }
    return std::pair<Value *, Value *>{carry_out, bitCast(sum)};
}

std::pair<Value *, Value *> IDISA_AVX2_Builder::bitblock_advance(Value * a, Value * shiftin, unsigned shift) {
    assert (a->getType() == mBitBlockType);
    if (shiftin->getType() == getInt8Ty() && shift == 1) {
        const uint32_t fw = mBitBlockWidth / 8;
        Type * const v32xi8Ty = FixedVectorType::get(getInt8Ty(), 32);
        Type * const v32xi32Ty = FixedVectorType::get(getInt32Ty(), 32);
        Value * shiftin_block = CreateInsertElement(Constant::getNullValue(v32xi8Ty), shiftin, (uint64_t) 0);
        shiftin_block = CreateShuffleVector(shiftin_block, UndefValue::get(v32xi8Ty), Constant::getNullValue(v32xi32Ty));
        shiftin_block = bitCast(shiftin_block);
        Value * field_shift = bitCast(mvmd_dslli(fw, a, shiftin_block, 1));
        Value * shifted = bitCast(CreateOr(CreateLShr(field_shift, fw - shift), CreateShl(a, shift)));
        Value * shiftout = hsimd_signmask(fw, a);
        shiftout = CreateTrunc(shiftout, getInt8Ty());
        shiftout = CreateAnd(shiftout, getInt8(0x80));
        return std::make_pair(shiftout, shifted);
    } else {
        return IDISA_SSE2_Builder::bitblock_advance(a, shiftin, shift);
    }
}

std::vector<Value *> IDISA_AVX2_Builder::simd_pext(unsigned fieldwidth, std::vector<Value *> v, Value * extract_mask) {
    if (hasFeature(Feature::AVX_BMI2)) {
        const auto n = getVectorBitWidth(v[0]) / fieldwidth;
        std::vector<Value *> mask(n);
        for (unsigned i = 0; i < n; i++) {
            mask[i] = mvmd_extract(fieldwidth, extract_mask, i);
        }
        std::vector<Value *> w(v.size());
        for (unsigned j = 0; j < v.size(); j++) {
            Value * result = UndefValue::get(v[j]->getType());
            for (unsigned i = 0; i < n; i++) {
                Value * v_i = mvmd_extract(fieldwidth, v[j], i);
                Value * bits = CreatePextract(v_i, mask[i]);
                result = mvmd_insert(fieldwidth, result, bits, i);
            }
            w[j] = fwCast(fieldwidth, result);
        }
        return w;
    }
    return IDISA_Builder::simd_pext(fieldwidth, v, extract_mask);
}

Value * IDISA_AVX2_Builder::simd_pdep(unsigned fieldwidth, Value * v, Value * deposit_mask) {
    if (hasFeature(Feature::AVX_BMI2)) {
        const auto n = getVectorBitWidth(v) / fieldwidth;
        Value * result = UndefValue::get(v->getType());
        for (unsigned i = 0; i < n; i++) {
            Value * v_i = mvmd_extract(fieldwidth, v, i);
            Value * mask_i = mvmd_extract(fieldwidth, deposit_mask, i);
            Value * bits = CreatePdeposit(v_i, mask_i);
            result = mvmd_insert(fieldwidth, result, bits, i);
        }
        return fwCast(fieldwidth, result);
    }
    return IDISA_Builder::simd_pdep(fieldwidth, v, deposit_mask);
}

std::pair<Value *, Value *> IDISA_AVX2_Builder::bitblock_indexed_advance(Value * strm, Value * index_strm, Value * shiftIn, unsigned shiftAmount) {
    const unsigned bitWidth = getSizeTy()->getBitWidth();
    if (hasFeature(Feature::AVX_BMI2) && ((bitWidth == 64) || (bitWidth == 32))) {
        Function * PEXT_f = (bitWidth == 64) ? Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pext_64)
                                          : Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pext_32);
        Function * PDEP_f = (bitWidth == 64) ? Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pdep_64)
                                          : Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pdep_32);
        Function * const popcount = Intrinsic::getDeclaration(getModule(), Intrinsic::ctpop, getSizeTy());
        Type * iBitBlock = getIntNTy(getBitBlockWidth());
        Value * shiftVal = getSize(shiftAmount);
        const auto n = getBitBlockWidth() / bitWidth;
        FixedVectorType * const vecTy = FixedVectorType::get(getSizeTy(), n);
        if (LLVM_LIKELY(shiftAmount < bitWidth)) {
            Value * carry = mvmd_extract(bitWidth, shiftIn, 0);
            Value * result = UndefValue::get(vecTy);
            for (unsigned i = 0; i < n; i++) {
                Value * s = mvmd_extract(bitWidth, strm, i);
                Value * ix = mvmd_extract(bitWidth, index_strm, i);
                Value * ix_popcnt = CreateCall(popcount->getFunctionType(), popcount, {ix});
                Value * bits = CreateCall(PEXT_f->getFunctionType(), PEXT_f, {s, ix});
                Value * adv = CreateOr(CreateShl(bits, shiftAmount), carry);
                // We have two cases depending on whether the popcount of the index pack is < shiftAmount or not.
                Value * popcount_small = CreateICmpULT(ix_popcnt, shiftVal);
                Value * carry_if_popcount_small =
                    CreateOr(CreateShl(bits, CreateSub(shiftVal, ix_popcnt)),
                                CreateLShr(carry, ix_popcnt));
                Value * carry_if_popcount_large = CreateLShr(bits, CreateSub(ix_popcnt, shiftVal));
                carry = CreateSelect(popcount_small, carry_if_popcount_small, carry_if_popcount_large);
                result = mvmd_insert(bitWidth, result, CreateCall(PDEP_f->getFunctionType(), PDEP_f, {adv, ix}), i);
            }
            Value * carryOut = mvmd_insert(bitWidth, allZeroes(), carry, 0);
            return std::pair<Value *, Value *>{bitCast(carryOut), bitCast(result)};
        } else if (shiftAmount <= mBitBlockWidth) {
            // The shift amount is always greater than the popcount of the individual
            // elements that we deal with.   This simplifies some of the logic.
            Value * carry = CreateBitCast(shiftIn, iBitBlock);
            Value * result = UndefValue::get(vecTy);
            for (unsigned i = 0; i < n; i++) {
                Value * s = mvmd_extract(bitWidth, strm, i);
                Value * ix = mvmd_extract(bitWidth, index_strm, i);
                Value * ix_popcnt = CreateCall(popcount->getFunctionType(), popcount, {ix});
                Value * bits = CreateCall(PEXT_f->getFunctionType(), PEXT_f, {s, ix});  // All these bits are shifted out (appended to carry).
                result = mvmd_insert(bitWidth, result, CreateCall(PDEP_f->getFunctionType(), PDEP_f, {mvmd_extract(bitWidth, carry, 0), ix}), i);
                carry = CreateLShr(carry, CreateZExt(ix_popcnt, iBitBlock)); // Remove the carry bits consumed, make room for new bits.
                carry = CreateOr(carry, CreateShl(CreateZExt(bits, iBitBlock), CreateZExt(CreateSub(shiftVal, ix_popcnt), iBitBlock)));
            }
            return std::pair<Value *, Value *>{bitCast(carry), bitCast(result)};
        } else {
            // The shift amount is greater than the total popcount.   We will consume popcount
            // bits from the shiftIn value only, and produce a carry out value of the selected bits.
            // elements that we deal with.   This simplifies some of the logic.
            Value * carry = CreateBitCast(shiftIn, iBitBlock);
            Value * result = UndefValue::get(vecTy);
            Value * carryOut = ConstantInt::getNullValue(iBitBlock);
            Value * generated = getSize(0);
            for (unsigned i = 0; i < n; i++) {
                Value * s = mvmd_extract(bitWidth, strm, i);
                Value * ix = mvmd_extract(bitWidth, index_strm, i);
                Value * ix_popcnt = CreateCall(popcount->getFunctionType(), popcount, {ix});
                Value * bits = CreateCall(PEXT_f->getFunctionType(), PEXT_f, {s, ix});  // All these bits are shifted out (appended to carry).
                result = mvmd_insert(bitWidth, result, CreateCall(PDEP_f->getFunctionType(), PDEP_f, {mvmd_extract(bitWidth, carry, 0), ix}), i);
                carry = CreateLShr(carry, CreateZExt(ix_popcnt, iBitBlock)); // Remove the carry bits consumed.
                carryOut = CreateOr(carryOut, CreateShl(CreateZExt(bits, iBitBlock), CreateZExt(generated, iBitBlock)));
                generated = CreateAdd(generated, ix_popcnt);
            }
            return std::pair<Value *, Value *>{bitCast(carryOut), bitCast(result)};
        }
    }
    return IDISA_Builder::bitblock_indexed_advance(strm, index_strm, shiftIn, shiftAmount);
}

Value * IDISA_AVX2_Builder::hsimd_signmask(unsigned fw, Value * a) {
    // AVX2 special cases
    if (mBitBlockWidth == 256) {
        if (fw == 8) {
            Function * signmask_f8func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx2_pmovmskb);
            Type * bitBlock_i8type = FixedVectorType::get(getInt8Ty(), mBitBlockWidth/8);
            Value * a_as_ps = CreateBitCast(a, bitBlock_i8type);
            return CreateCall(signmask_f8func->getFunctionType(), signmask_f8func, a_as_ps);
        }
    }
    // Otherwise use default SSE logic.
    return IDISA_AVX_Builder::hsimd_signmask(fw, a);
}

Value * IDISA_AVX2_Builder::mvmd_srl(unsigned fw, Value * a, Value * shift, const bool safe) {
    // Intrinsic::x86_avx2_permd) allows an efficient implementation for field width 32.
    // Translate larger field widths to 32 bits.
    if (LLVM_LIKELY(mBitBlockWidth == 256)) {
        if (fw > 32) {
            return fwCast(fw, mvmd_srl(32, a, CreateMul(shift, ConstantInt::get(shift->getType(), fw/32)), safe));
        } else if (fw == 32) {
            Function * permuteFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx2_permd);
            const unsigned fieldCount = mBitBlockWidth/fw;
            Type * fieldTy = getIntNTy(fw);
            SmallVector<Constant *, 16> indexes(fieldCount);
            for (unsigned int i = 0; i < fieldCount; i++) {
                indexes[i] = ConstantInt::get(fieldTy, i);
            }
            Constant * indexVec = ConstantVector::get(indexes);
            Constant * fieldCountSplat = getSplat(fieldCount, ConstantInt::get(fieldTy, fieldCount));
            Value * shiftSplat = simd_fill(fw, CreateZExtOrTrunc(shift, fieldTy));
            Value * permuteVec = CreateAdd(indexVec, shiftSplat);
            // Zero out fields that are above the max.
            permuteVec = simd_and(permuteVec, simd_ult(fw, permuteVec, fieldCountSplat));
            // Insert a zero value at position 0 (OK for shifts > 0)
            Value * a0 = mvmd_insert(fw, a, Constant::getNullValue(fieldTy), 0);
            Value * shifted = CreateCall(permuteFunc->getFunctionType(), permuteFunc, {a0, permuteVec});
            return fwCast(32, simd_if(1, simd_eq(fw, shiftSplat, allZeroes()), a, shifted));
        }
    }
    return IDISA_AVX_Builder::mvmd_srl(fw, a, shift, safe);
}

Value * IDISA_AVX2_Builder::mvmd_sll(unsigned fw, Value * a, Value * shift, const bool safe) {
    // Intrinsic::x86_avx2_permd) allows an efficient implementation for field width 32.
    // Translate larger field widths to 32 bits.
    if (mBitBlockWidth == 256) {
        if (fw > 32) {
            return fwCast(fw, mvmd_sll(32, a, CreateMul(shift, ConstantInt::get(shift->getType(), fw/32)), safe));
        } else if (fw == 32) {
            Function * permuteFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx2_permd);
            const unsigned fieldCount = mBitBlockWidth/fw;
            Type * fieldTy = getIntNTy(fw);
            SmallVector<Constant *, 16> indexes(fieldCount);
            for (unsigned int i = 0; i < fieldCount; i++) {
                indexes[i] = ConstantInt::get(fieldTy, i);
            }
            Constant * indexVec = ConstantVector::get(indexes);
            Value * shiftSplat = simd_fill(fw, CreateZExtOrTrunc(shift, fieldTy));
            Value * permuteVec = CreateSub(indexVec, shiftSplat);
            // Negative indexes are for fields that must be zeroed.  Convert the
            // permute constant to an all ones value, that will select item 7.
            permuteVec = simd_or(permuteVec, simd_lt(fw, permuteVec, fwCast(fw, allZeroes())));
            // Insert a zero value at position 7 (OK for shifts > 0)
            Value * a0 = mvmd_insert(fw, a, Constant::getNullValue(fieldTy), 7);
            Value * shifted = CreateCall(permuteFunc->getFunctionType(), permuteFunc, {a0, permuteVec});
            return fwCast(32, simd_if(1, simd_eq(fw, shiftSplat, allZeroes()), a, shifted));
        }
    }
    return IDISA_Builder::mvmd_sll(fw, a, shift, safe);
}


Value * IDISA_AVX2_Builder::mvmd_shuffle(unsigned fw, Value * a, Value * index_vector) {
    if (mBitBlockWidth == 256) {
        if (fw == 64) {
#if 0
            const unsigned fieldCount = mBitBlockWidth / fw;
            // Create a table for shuffling with smaller field widths.
            Constant * idxMask = getSplat(fieldCount, ConstantInt::get(getIntNTy(fw), fieldCount-1));
            Value * idx = simd_and(index_vector, idxMask);
            const auto half_fw = fw/2;
            const auto field_count = mBitBlockWidth/half_fw;
            // Build a ConstantVector of alternating 0 and 1 values.
            SmallVector<Constant *, 16> Idxs(field_count);
            for (unsigned int i = 0; i < field_count; i++) {
                Idxs[i] = ConstantInt::get(getIntNTy(fw/2), i & 1);
            }
            Constant * splat01 = ConstantVector::get(Idxs);
            Value * half_fw_indexes = simd_or(idx, mvmd_slli(half_fw, idx, 1));
            half_fw_indexes = simd_add(fw, simd_add(fw, half_fw_indexes, half_fw_indexes), splat01);
            return fwCast(fw, mvmd_shuffle(half_fw, a, half_fw_indexes));
#else
            constexpr auto fieldCount = 256 / 64;
            Value * A = CreateMul(fwCast(64, index_vector), getSplat(fieldCount, getInt64(0x0000000200000002ULL)));
            index_vector = CreateOr(A, getSplat(fieldCount, getInt64(0x0000000100000000ULL)));
            return fwCast(64, mvmd_shuffle(32, a, index_vector));
#endif
        } else if (fw == 32) {
            Function * shuf32Func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx2_permd);
            return CreateCall(shuf32Func->getFunctionType(), shuf32Func, {fwCast(32, a), fwCast(32, index_vector)});
        } else if (fw == 16) {
            // TODO: if we can use vpshuflw and vpshufhw, we may be able to do this a bit more efficiently
            // but LLVM doesn't seem to have intrinsics for them? Check whether shufflevector can deduce them.
            // For now, we simply transform the 16-bit indices into pairs of adjacent 8-bit ones
            constexpr auto fieldCount = 256 / 16;
            Value * A = CreateMul(fwCast(16, index_vector), getSplat(fieldCount, getInt16(0x0202)));
            index_vector = CreateOr(A, getSplat(fieldCount, getInt16(0x0100)));
            return fwCast(16, mvmd_shuffle(8, fwCast(8, a), fwCast(8, index_vector)));
        } else if (fw == 8) {
            constexpr unsigned fieldCount = 256 / 8;

            IntegerType * const int8Ty = getInt8Ty();

            Constant * SIXTEEN = getSplat(fieldCount, ConstantInt::get(int8Ty, 16));

            auto createShuffleVec = [&](int a, int b, int c, int d) {
                FixedArray<Constant *, 4> idx;
                idx[0] = getInt32(a);
                idx[1] = getInt32(b);
                idx[2] = getInt32(c);
                idx[3] = getInt32(d);
                return ConstantVector::get(idx);
            };

            // TODO: I want to call vpermq directly since LLVM 15 shufflevector doesn't produce it but LLVM doesn't
            // seem to support the instruction or intrinsic? check if later versions do.

            // Function * permuteFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx2_permq);

            FixedVectorType * vec64Ty = FixedVectorType::get(getInt64Ty(), 256 / 64);
            Function * shufFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx2_pshuf_b);
            Value * const a64 = CreateBitCast(a, vec64Ty);
            FixedVectorType * vecTy = FixedVectorType::get(int8Ty, 256 / 8);
            index_vector = CreateBitCast(index_vector, vecTy);

            FixedArray<Value *, 2> args;
            Value * a0 = CreateShuffleVector(a64, UndefValue::get(vec64Ty), createShuffleVec(0, 1, 0, 1));
            args[0] = CreateBitCast(a0, vecTy);
            args[1] = CreateOr(index_vector, CreateSExt(CreateICmpUGE(index_vector, SIXTEEN), vecTy));
            Value * a1 = CreateCall(shufFunc->getFunctionType(), shufFunc, args);
            assert (a1->getType() == vecTy);
            Value * b0 = CreateShuffleVector(a64, UndefValue::get(vec64Ty), createShuffleVec(2, 3, 2, 3));
            assert (b0->getType() == vec64Ty);
            args[0] = CreateBitCast(b0, vecTy);
            args[1] = CreateSub(index_vector, SIXTEEN); // sets sign bit automatically if selected in a0
            Value * b1 = CreateCall(shufFunc->getFunctionType(), shufFunc, args);
            assert (b1->getType() == vecTy);
            return CreateOr(a1, b1);
        }
    }
    return IDISA_Builder::mvmd_shuffle(fw, a, index_vector);
}

llvm::Value * IDISA_AVX2_Builder::mvmd_shuffle2(unsigned fw, llvm::Value * table0, llvm::Value * table1, llvm::Value * index_vector) {
    return IDISA_Builder::mvmd_shuffle2(fw, table0, table1, index_vector);
}

Value * IDISA_AVX2_Builder::mvmd_compress(unsigned fw, Value * a, Value * select_mask) {
    if (hasFeature(Feature::AVX_BMI2) && (mBitBlockWidth == 256)) {

        if (fw == 64) {
            Function * PDEP_func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pdep_32);
            Value * mask = CreateZExtOrTrunc(select_mask, getInt32Ty());
            Value * mask32 = CreateMul(CreateCall(PDEP_func->getFunctionType(), PDEP_func, {mask, getInt32(0x55)}), getInt32(3));
            Value * result = fwCast(fw, mvmd_compress(32, fwCast(32, a), CreateTrunc(mask32, getInt8Ty())));
            return result;
        } else if (fw == 32) {
            Type * v1xi32Ty = FixedVectorType::get(getInt32Ty(), 1);
            Type * v8xi32Ty = FixedVectorType::get(getInt32Ty(), 8);
            Type * v8xi1Ty = FixedVectorType::get(getInt1Ty(), 8);
            Constant * mask0000000Fsplaat = getSplat(8, ConstantInt::get(getInt32Ty(), 0xF));
            Function * PEXT_func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pext_32);
            Function * PDEP_func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pdep_32);
            Function * const popcount_func = Intrinsic::getDeclaration(getModule(), Intrinsic::ctpop, getInt32Ty());
            // First duplicate each mask bit to select 4-bit fields
            Value * mask = CreateZExtOrTrunc(select_mask, getInt32Ty());
            Value * field_count = CreateCall(popcount_func->getFunctionType(), popcount_func, mask);
            assert (field_count->getType()->getIntegerBitWidth() >= 32);
            Value * spread = CreateCall(PDEP_func->getFunctionType(), PDEP_func, {mask, getInt32(0x11111111)});
            Value * ext_mask = CreateMul(spread, getInt32(0xF));
            // Now extract the 4-bit index values for the required fields.
            Value * indexes = CreateCall(PEXT_func->getFunctionType(), PEXT_func, {getInt32(0x76543210), ext_mask});
            // Broadcast to all fields
            Value * bdcst = CreateShuffleVector(CreateBitCast(indexes, v1xi32Ty),
                                                UndefValue::get(v1xi32Ty),
                                                ConstantVector::getNullValue(v8xi32Ty));
            Constant * Shifts[8];
            for (unsigned i = 0; i < 8; i++) {
                Shifts[i] = getInt32(i * 4);
            }
            Value * shuf = CreateAnd(CreateLShr(bdcst, ConstantVector::get({Shifts, 8})), mask0000000Fsplaat);

            Value * compress = mvmd_shuffle(32, a, shuf);
            Value * field_mask = CreateTrunc(CreateSub(CreateShl(getInt32(1), field_count), getInt32(1)), getInt8Ty());
            return CreateAnd(compress, CreateSExt(CreateBitCast(field_mask, v8xi1Ty), v8xi32Ty));
        } else if (fw >= 8) {

            unsigned fieldCount = 256 / fw;

            // Step 1: Initialize indices as 6-bit bixnum in an array of 64-bit integers
            uint64_t indices[6] = {
                0xAAAAAAAAAAAAAAAA,
                0xCCCCCCCCCCCCCCCC,
                0xF0F0F0F0F0F0F0F0,
                0xFF00FF00FF00FF00,
                0xFFFF0000FFFF0000,
                0xFFFFFFFF00000000
            };

            // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected
            Function * pextFunc = nullptr;
            if (LLVM_LIKELY(fieldCount == 64)) {
                pextFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pext_64);
            } else {
                assert (fieldCount <= 32);
                pextFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pext_32);
                if (LLVM_UNLIKELY(fieldCount < 32)) {
                    assert (select_mask->getType()->getIntegerBitWidth() <= 32);
                    select_mask = CreateZExt(select_mask, getInt32Ty());
                }
            }

            // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected

            Value * permute_vec = nullptr;
            const auto m = floor_log2(fieldCount);
            FixedArray<Value *, 2> args;
            args[1] = select_mask;

            IntegerType * intTy = getIntNTy((fieldCount == 64) ? 64 : 32);

            Type * resultTy = fwVectorType(fw);

            for (unsigned i = 0; i < m; ++i) {
                args[0] = ConstantInt::get(intTy, indices[i]);
                Value * const expanded = CreateCall(pextFunc->getFunctionType(), pextFunc, args);
                Value * byteExpanded = esimd_bitspread(fw, expanded);
                if (i == 0) {
                    permute_vec = byteExpanded;
                } else {
                    Value * shiftedExpand = simd_slli(64, byteExpanded, i);
                    permute_vec = simd_or(permute_vec, shiftedExpand);
                }
            }
            // // Step 4: Use mvmd_shuffle2 to shuffle using permute_vec
            Value * const shuffled = mvmd_shuffle(fw, a, permute_vec);
            Value * const count = CreatePopcount(select_mask);
            Constant * ALL_ONES = ConstantVector::getAllOnesValue(resultTy);
            Value * mask = CreateNot(mvmd_sll(fw, ALL_ONES, count));
            mask = CreateSelect(CreateICmpNE(count, ConstantInt::get(intTy, fieldCount)), mask, ALL_ONES);
            assert (shuffled->getType() == mask->getType());
            return CreateAnd(shuffled, mask);
        }
    }

    return IDISA_Builder::mvmd_compress(fw, a, select_mask);
}

Value * IDISA_AVX2_Builder::mvmd_expand(unsigned fw, Value * a, Value * select_mask) {
    if (hasFeature(Feature::AVX_BMI2) && (mBitBlockWidth == 256)) {
         if (fw >= 8) {

             const auto fieldCount = 256 / fw;

             uint64_t indices[6] = {
                 0xAAAAAAAAAAAAAAAA,
                 0xCCCCCCCCCCCCCCCC,
                 0xF0F0F0F0F0F0F0F0,
                 0xFF00FF00FF00FF00,
                 0xFFFF0000FFFF0000,
                 0xFFFFFFFF00000000
             };

             Function * pdepFunc = nullptr;
             if (LLVM_LIKELY(fieldCount == 64)) {
                 pdepFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pdep_64);
             } else {
                 assert (fieldCount <= 32);
                 pdepFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pdep_32);
                 if (LLVM_UNLIKELY(fieldCount < 32)) {
                     assert (select_mask->getType()->getIntegerBitWidth() <= 32);
                     select_mask = CreateZExt(select_mask, getInt32Ty());
                 }
             }

             // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected

             Value * permute_vec = nullptr;
             const auto m = floor_log2(fieldCount);
             FixedArray<Value *, 2> args;
             args[1] = select_mask;

             IntegerType * intTy = getIntNTy((fieldCount == 64) ? 64 : 32);
             for (unsigned i = 0; i < m; ++i) {
                 args[0] = ConstantInt::get(intTy, indices[i]);
                 Value * const expanded = CreateCall(pdepFunc->getFunctionType(), pdepFunc, args);
                 Value * byteExpanded = esimd_bitspread(fw, expanded);
                 if (i == 0) {
                     permute_vec = byteExpanded;
                 } else {
                     Value * shiftedExpand = simd_slli(64, byteExpanded, i);
                     permute_vec = simd_or(permute_vec, shiftedExpand);
                 }
             }

             // // Step 4: Use mvmd_shuffle to shuffle using permute_vec
             Value * const shuffled = mvmd_shuffle(fw, a, permute_vec);
             Value * const mask = simd_any(fw, esimd_bitspread(fw, select_mask));
             assert (shuffled->getType() == mask->getType());
             return CreateAnd(shuffled, mask);
         }
    }
    return IDISA_AVX_Builder::mvmd_expand(fw, a, select_mask);
}


std::string IDISA_AVX512F_Builder::getBuilderUniqueName() {
    return mBitBlockWidth != 512 ? "AVX512F_" + std::to_string(mBitBlockWidth) : "AVX512BW";
}

Value * IDISA_AVX512F_Builder::hsimd_packh(unsigned fw, Value * a, Value * b) {
    if ((mBitBlockWidth == 512) && (fw == 16)) {
        a = fwCast(fw, a);
        a = IDISA_Builder::simd_srli(fw, a, fw/2);
        b = fwCast(fw, b);
        b = IDISA_Builder::simd_srli(fw, b, fw/2);
        return hsimd_packl(fw, a, b);
    }
    return IDISA_Builder::hsimd_packh(fw, a, b);
}

Value * IDISA_AVX512F_Builder::hsimd_packl(unsigned fw, Value * a, Value * b) {
    if ((mBitBlockWidth == 512) && (fw == 16)) {

        const unsigned int field_count = 64;
        Constant * Idxs[field_count];
        for (unsigned int i = 0; i < field_count; i++) {
            Idxs[i] = getInt32(i);
        }
        Constant * shuffleMask = ConstantVector::get({Idxs, 64});
        Value * a1 = CreateTrunc(fwCast(fw, a), FixedVectorType::get(getInt8Ty(), 32));
        Value * b1 = CreateTrunc(fwCast(fw, b), FixedVectorType::get(getInt8Ty(), 32));
        return CreateShuffleVector(a1, b1, shuffleMask);
    }
    return IDISA_Builder::hsimd_packl(fw, a, b);
}

Value * IDISA_AVX512F_Builder::hsimd_packus(unsigned fw, Value * a, Value * b) {
    if (hasFeature(Feature::AVX512_BW) && ((fw == 16) || (fw == 32)) && (getVectorBitWidth(a) == AVX512_width)) {
        Function * pack_func = Intrinsic::getDeclaration(getModule(), fw == 16 ? Intrinsic::x86_avx512_packuswb_512 : Intrinsic::x86_avx512_packusdw_512);
        Value * packed = CreateCall(pack_func->getFunctionType(), pack_func, {fwCast(fw, a), fwCast(fw, b)});
        auto field_count = AVX512_width/64;
        SmallVector<Constant *, 16> Idxs(field_count);
        for (unsigned int i = 0; i < field_count/2; i++) {
            Idxs[i] = getInt32(2*i);
            Idxs[i + field_count/2] = getInt32(2*i + 1);
        }
        Constant * shuffleMask = ConstantVector::get(Idxs);
        return CreateShuffleVector(fwCast(64, packed), UndefValue::get(fwVectorType(64)), shuffleMask);
    }
    // Otherwise use default logic.
    return IDISA_Builder::hsimd_packus(fw, a, b);
}

Value * IDISA_AVX512F_Builder::hsimd_packss(unsigned fw, Value * a, Value * b) {
    if (hasFeature(Feature::AVX512_BW) && ((fw == 16) || (fw == 32)) && (getVectorBitWidth(a) == AVX512_width)) {
        Function * pack_func = Intrinsic::getDeclaration(getModule(), fw == 16 ? Intrinsic::x86_avx512_packsswb_512 : Intrinsic::x86_avx512_packssdw_512);
        Value * packed = CreateCall(pack_func->getFunctionType(), pack_func, {fwCast(fw, a), fwCast(fw, b)});
        auto field_count = AVX512_width/64;
        SmallVector<Constant *, 16> Idxs(field_count);
        for (unsigned int i = 0; i < field_count/2; i++) {
            Idxs[i] = getInt32(2*i);
            Idxs[i + field_count/2] = getInt32(2*i + 1);
        }
        Constant * shuffleMask = ConstantVector::get(Idxs);
        return CreateShuffleVector(fwCast(64, packed), UndefValue::get(fwVectorType(64)), shuffleMask);
    }
    // Otherwise use default logic.
    return IDISA_Builder::hsimd_packus(fw, a, b);
}


Value * IDISA_AVX512F_Builder::esimd_bitspread(unsigned fw, Value * bitmask) {
    const auto field_count = mBitBlockWidth / fw;
    Type * maskTy = FixedVectorType::get(getInt1Ty(), field_count);
    Type * resultTy = fwVectorType(fw);
    return CreateZExt(CreateBitCast(CreateZExtOrTrunc(bitmask, getIntNTy(field_count)), maskTy), resultTy);
}

Value * IDISA_AVX512F_Builder::mvmd_srl(unsigned fw, Value * a, Value * shift, const bool safe) {
    if (mBitBlockWidth == 512 && fw >= 8) {
        const auto fieldCount = 512 / fw;
        Type * fieldTy = getIntNTy(fw);
        SmallVector<Constant *, 64> indexes(fieldCount);
        for (unsigned i = 0; i < fieldCount; i++) {
            indexes[i] = ConstantInt::get(fieldTy, i);
        }
        Constant * indexVec = ConstantVector::get(indexes);
        Value * broadcast = simd_fill(fw, CreateZExtOrTrunc(shift, fieldTy));
        Value * permuteVec = CreateAdd(indexVec, broadcast);
        return mvmd_shuffle2(fw, fwCast(fw, a), fwCast(fw, allZeroes()), permuteVec);
    }
    return IDISA_Builder::mvmd_srl(fw, a, shift, safe);
}

Value * IDISA_AVX512F_Builder::mvmd_sll(unsigned fw, Value * a, Value * shift, const bool safe) {
    if (mBitBlockWidth == 512 && fw >= 8) {
        const auto fieldCount = 512 / fw;
        Type * fieldTy = getIntNTy(fw);
        SmallVector<Constant *, 64> indexes(fieldCount);
        for (unsigned i = 0; i < fieldCount; i++) {
            indexes[i] = ConstantInt::get(fieldTy, fieldCount + i);
        }
        Constant * indexVec = ConstantVector::get(indexes);
        Value * broadcast = simd_fill(fw, CreateZExtOrTrunc(shift, fieldTy));
        Value * permuteVec = CreateSub(indexVec, broadcast);
        return mvmd_shuffle2(fw, fwCast(fw, allZeroes()), fwCast(fw, a), permuteVec);
    }
    return IDISA_Builder::mvmd_sll(fw, a, shift);
}

Value * IDISA_AVX512F_Builder::mvmd_shuffle(unsigned fw, Value * data_table, Value * index_vector) {
    if (mBitBlockWidth == 512 && ((fw == 64) || (fw == 32) | (fw == 16))) {
        return mvmd_shuffle2(fw, data_table, data_table, index_vector);
    }
    return IDISA_Builder::mvmd_shuffle(fw, data_table, index_vector);
}


#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(7, 0, 0)
#define AVX512_MASK_PERMUTE_INTRINSIC(i) Intrinsic::x86_avx512_mask_vpermt2##i
#else
#define AVX512_MASK_PERMUTE_INTRINSIC(i) Intrinsic::x86_avx512_vpermi2##i
#endif

Value * IDISA_AVX512F_Builder::mvmd_shuffle2(unsigned fw, Value * table0, Value * table1, Value * index_vector) {
    if (mBitBlockWidth == 512) {
        Function * permuteFunc = nullptr;
        if (fw == 32) {
            permuteFunc = Intrinsic::getDeclaration(getModule(), AVX512_MASK_PERMUTE_INTRINSIC(var_d_512));
        } else if (fw == 64) {
            permuteFunc = Intrinsic::getDeclaration(getModule(), AVX512_MASK_PERMUTE_INTRINSIC(var_q_512));
        } else if (fw == 16 && hasFeature(Feature::AVX512_BW)) {
            permuteFunc = Intrinsic::getDeclaration(getModule(), AVX512_MASK_PERMUTE_INTRINSIC(var_hi_512));
        } else if (fw == 8) {
            if (hasFeature(Feature::AVX512_VBMI)) {
                permuteFunc = Intrinsic::getDeclaration(getModule(), AVX512_MASK_PERMUTE_INTRINSIC(var_qi_512));
            } else if (hasFeature(Feature::AVX512_BW)) {

                // If we have AVX512BW but not AVX512VBMI, we can use 16 bit shuffles to replicate an 8 bit shuffle.
                // This requires us to split the table look up into a lower and higher "half" table and index vectors
                // since we need to zero extend each field.

                // Although the tables can be easily zero extended, the indices are a bit trickier since we could have:
                // <0, 63, 1, 62, ...> as a pattern. Thus we build up both the results from selecting the lower and higher
                // tables separately then OR them together.


                VectorType * vty = fwVectorType(8);

                assert (index_vector->getType() == vty);

                Constant * const ZEROES = ConstantVector::getNullValue(vty);

                #define ZEXT16L(T) esimd_mergel(8, (T), ZEROES)
                #define ZEXT16H(T) esimd_mergeh(8, (T), ZEROES)

                Constant * const ALL_64 = getSplat(512 / 8, getInt8(64));
                Value * const InL = simd_lt(8, index_vector, ALL_64);
                assert (InL->getType() == vty);
                Value * IndexVectorL = CreateAnd(index_vector, InL);
                Value * const InH = CreateNot(InL);
                Value * IndexVectorH = CreateAnd(CreateSub(index_vector, ALL_64), InH);
                Value * IndexVector = CreateOr(IndexVectorL, IndexVectorH);
                Value * const IL = ZEXT16L(IndexVector);
                Value * const IH = ZEXT16H(IndexVector);
                auto SelectFromHalfTable = [&](Value * table, Value * Mask) {
                    Value * T0 = ZEXT16L(table);
                    Value * T1 = ZEXT16H(table);
                    Value * const A = mvmd_shuffle2(16, T0, T1, IL);
                    Value * const B = mvmd_shuffle2(16, T0, T1, IH);
                    Value * packed = fwCast(8, hsimd_packl(16, A, B));
                    return CreateAnd(packed, Mask);
                };
                #undef ZEXT16L
                #undef ZEXT16H

                Value * const L = SelectFromHalfTable(table0, InL);
                assert (L->getType() == vty);
                Value * const H = SelectFromHalfTable(table1, InH);
                assert (H->getType() == vty);
                return CreateOr(L, H);
            }
        }

        if (permuteFunc) {
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(7, 0, 0)
            const unsigned fieldCount = mBitBlockWidth/fw;
            Constant * mask = ConstantInt::getAllOnesValue(getIntNTy(fieldCount));
            return CreateCall(permuteFunc->getFunctionType(), permuteFunc, {fwCast(fw, index_vector), fwCast(fw, table0), fwCast(fw, table1), mask});
#else
            return CreateCall(permuteFunc->getFunctionType(), permuteFunc, {fwCast(fw, table0), fwCast(fw, index_vector), fwCast(fw, table1)});
#endif
        }
    }
    return IDISA_Builder::mvmd_shuffle2(fw, table0, table1, index_vector);
}


#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(9, 0, 0)
#define AVX512_MASK_COMPRESS_INTRINSIC_64 Intrinsic::x86_avx512_mask_compress_q_512
#define AVX512_MASK_COMPRESS_INTRINSIC_32 Intrinsic::x86_avx512_mask_compress_d_512
#else
#define AVX512_MASK_COMPRESS_INTRINSIC_64 Intrinsic::x86_avx512_mask_compress
#define AVX512_MASK_COMPRESS_INTRINSIC_32 Intrinsic::x86_avx512_mask_compress
#endif

Value * IDISA_AVX512F_Builder::mvmd_compress(unsigned fw, Value * a, Value * select_mask) {
    unsigned fieldCount = mBitBlockWidth / fw;
    Value * mask = CreateZExtOrTrunc(select_mask, getIntNTy(fieldCount));
    if (mBitBlockWidth == 512 && fw == 32) {
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(9, 0, 0)
        Function * compressFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_compress_d_512);
        return CreateCall(compressFunc->getFunctionType(), compressFunc, {fwCast(32, a), fwCast(32, allZeroes()), mask});
#else
        Type * maskTy = FixedVectorType::get(getInt1Ty(), fieldCount);
        Function * compressFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_compress, fwVectorType(fw));
        return CreateCall(compressFunc->getFunctionType(), compressFunc, {fwCast(32, a), fwCast(32, allZeroes()), CreateBitCast(mask, maskTy)});
#endif
    }
    if (mBitBlockWidth == 512 && fw == 64) {
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(9, 0, 0)
        Function * compressFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_compress_q_512);
        return CreateCall(compressFunc->getFunctionType(), compressFunc, {fwCast(64, a), fwCast(64, allZeroes()), mask});
#else
        Type * maskTy = FixedVectorType::get(getInt1Ty(), fieldCount);
        Function * compressFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_compress, fwVectorType(fw));
        return CreateCall(compressFunc->getFunctionType(), compressFunc, {fwCast(64, a), fwCast(64, allZeroes()), CreateBitCast(mask, maskTy)});
#endif
    }

    if (mBitBlockWidth == 512 && fw == 8) {
        if (hasFeature(Feature::AVX512_VBMI2)){
 #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(9, 0, 0)
            Function * compressFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_compress_b_512);
            return CreateCall(compressFunc->getFunctionType(), compressFunc, {fwCast(8, a), fwCast(8, allZeroes()), mask});
 #else
            Type * maskTy = FixedVectorType::get(getInt1Ty(), fieldCount);
            Function * compressFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_compress, fwVectorType(fw));
            return CreateCall(compressFunc->getFunctionType(), compressFunc, {fwCast(8, a), fwCast(8, allZeroes()), CreateBitCast(mask, maskTy)});
 #endif
        } else if (hasFeature(Feature::AVX512_VBMI) || hasFeature(Feature::AVX512_BW)) {

            // Step 1: Initialize indices as 6-bit bixnum in an array of 64-bit integers
            uint64_t indices[6] = {
                0xAAAAAAAAAAAAAAAA,
                0xCCCCCCCCCCCCCCCC,
                0xF0F0F0F0F0F0F0F0,
                0xFF00FF00FF00FF00,
                0xFFFF0000FFFF0000,
                0xFFFFFFFF00000000
            };

            // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected
            Function * pextFunc = nullptr;
            if (LLVM_LIKELY(fieldCount == 64)) {
                pextFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pext_64);
            } else {
                assert (fieldCount <= 32);
                pextFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pext_32);
                if (LLVM_UNLIKELY(fieldCount < 32)) {
                    mask = CreateZExt(mask, getInt32Ty());
                }
            }

            // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected

            Value * permute_vec = nullptr;
            const auto m = floor_log2(fieldCount);
            FixedArray<Value *, 2> args;
            args[1] = select_mask;

            IntegerType * intTy = getIntNTy((fieldCount == 64) ? 64 : 32);

            Type * vTy = fwVectorType(fw);

            for (unsigned i = 0; i < m; ++i) {
                args[0] = ConstantInt::get(intTy, indices[i]);
                Value * const expanded = CreateCall(pextFunc->getFunctionType(), pextFunc, args);
                Value * byteExpanded = esimd_bitspread(fw, expanded);
                if (i == 0) {
                    permute_vec = byteExpanded;
                } else {
                    Value * shiftedExpand = simd_slli(64, byteExpanded, i);
                    permute_vec = simd_or(permute_vec, shiftedExpand);
                }
            }

            // // Step 4: Use mvmd_shuffle2 to shuffle using permute_vec
            Constant * zero_vec = ConstantVector::getNullValue(vTy);
            Value * const shuffled = mvmd_shuffle2(fw, a, zero_vec, CreateBitCast(permute_vec, vTy));
            Value * const count = CreatePopcount(CreateZExt(select_mask, intTy));
            assert (count->getType() == intTy);
            // try a bitspread + any? check ASM
            Constant * ALL_ONES = ConstantVector::getAllOnesValue(vTy);
            Value * mask = CreateNot(mvmd_sll(fw, ALL_ONES, count, false));
            mask = CreateSelect(CreateICmpNE(count, ConstantInt::get(intTy, fieldCount)), mask, ALL_ONES);
            assert (shuffled->getType() == mask->getType());
            return CreateAnd(shuffled, mask);
        }

     }

    return IDISA_AVX2_Builder::mvmd_compress(fw, a, select_mask);
}

#define MVMD_EXPAND_BY_INDUCTIVE_DOUBLING
Value * IDISA_AVX512F_Builder::mvmd_expand(unsigned fw, Value * a, Value * select_mask) {
    const auto fieldCount = mBitBlockWidth / fw;
    Value * mask = CreateZExtOrTrunc(select_mask, getIntNTy(fieldCount));
    bool has_avx_512_mask_expand = (fw == 32) || (fw == 64) || (hasFeature(Feature::AVX512_VBMI2) && ((fw == 16) | (fw == 8)));
    if (has_avx_512_mask_expand) {
        Type * maskTy = FixedVectorType::get(getInt1Ty(), fieldCount);
        Function * expandFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_expand, fwVectorType(fw));
        return CreateCall(expandFunc->getFunctionType(), expandFunc, {fwCast(fw, a), fwCast(fw, allZeroes()), CreateBitCast(mask, maskTy)});
    } else if ((fw == 8) || (fw == 16)) {
#ifdef MVMD_EXPAND_BY_INDUCTIVE_DOUBLING
        Value * hi_mask = CreateLShr(mask, Constant::getIntegerValue(mask->getType(), APInt(fieldCount, fieldCount/2)));
        hi_mask = CreateTrunc(hi_mask, getIntNTy(fieldCount/2));
        Value * lo_mask = CreateTrunc(mask, getIntNTy(fieldCount/2));
        Value * lo_to_hi = CreatePopcount(CreateNot(lo_mask));
        Value * lo_vec = esimd_mergel(fw, a, allZeroes());
        Value * hi_vec = esimd_mergeh(fw, mvmd_sll(fw, a, lo_to_hi, true), allZeroes());
        Value * expand_lo = mvmd_expand(fw * 2, lo_vec, lo_mask);
        Value * expand_hi = mvmd_expand(fw * 2, hi_vec, hi_mask);
        Value * packed = bitCast(hsimd_packl(fw * 2, expand_lo, expand_hi));
        return packed;
#else
        uint64_t indices[6] = {
            0xAAAAAAAAAAAAAAAA,
            0xCCCCCCCCCCCCCCCC,
            0xF0F0F0F0F0F0F0F0,
            0xFF00FF00FF00FF00,
            0xFFFF0000FFFF0000,
            0xFFFFFFFF00000000
        };

        Function * pdepFunc = nullptr;
        if (LLVM_LIKELY(fieldCount == 64)) {
            pdepFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pdep_64);
        } else {
            assert (fieldCount <= 32);
            pdepFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_bmi_pdep_32);
            if (LLVM_UNLIKELY(fieldCount < 32)) {
                mask = CreateZExt(mask, getInt32Ty());
            }
        }

        // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected

        Value * permute_vec = nullptr;
        const auto m = floor_log2(fieldCount);
        FixedArray<Value *, 2> args;
        args[1] = mask;
        const auto popFW = (fieldCount == 64) ? 64U : 32U;
        for (unsigned i = 0; i < m; ++i) {
            args[0] = getIntN(popFW, indices[i]);
            Value * const expanded = CreateCall(pdepFunc->getFunctionType(), pdepFunc, args);
            assert (expanded->getType()->getIntegerBitWidth() == popFW);
            Value * byteExpanded = esimd_bitspread(fw, expanded);
            if (i == 0) {
                permute_vec = byteExpanded;
            } else {
                Value * shiftedExpand = simd_slli(64, byteExpanded, i);
                permute_vec = simd_or(permute_vec, shiftedExpand);
            }
        }

        a = fwCast(fw, a);
        permute_vec = fwCast(fw, permute_vec);

        // // Step 4: Use mvmd_shuffle2 to shuffle using permute_vec
        Constant * zero_vec = ConstantVector::getNullValue(a->getType());
        Value * const shuffled = mvmd_shuffle2(fw, a, zero_vec, permute_vec);
        return CreateAnd(shuffled, simd_any(fw, esimd_bitspread(fw, mask)));
#endif
    }
    return IDISA_AVX2_Builder::mvmd_expand(fw, a, select_mask);
}

Value * IDISA_AVX512F_Builder:: mvmd_slli(unsigned fw, Value * a, unsigned shift) {
    if (shift == 0) return a;
    if (fw > 32) {
        return fwCast(fw, mvmd_slli(32, a, shift * (fw/32)));
    } else if (((shift % 2) == 0) && (fw < 32)) {
        return fwCast(fw, mvmd_slli(2 * fw, a, shift / 2));
    }
    if ((fw == 32) || (hasFeature(Feature::AVX512_BW) && (fw == 16)))   {
        return mvmd_dslli(fw, a, allZeroes(), shift);
    } else {
        unsigned field32_shift = (shift * fw) / 32;
        unsigned bit_shift = (shift * fw) % 32;
        Value * const L = simd_slli(32, mvmd_slli(32, a, field32_shift), bit_shift);
        Value * const R = simd_srli(32, mvmd_slli(32, a, field32_shift + 1), 32-bit_shift);
        assert (L->getType() == R->getType());
        return fwCast(fw, CreateOr(L, R));
    }
}

Value * IDISA_AVX512F_Builder:: mvmd_dslli(unsigned fw, Value * a, Value * b, unsigned shift) {
    if (shift == 0) return a;
    if (fw > 32) {
        return fwCast(fw, mvmd_dslli(32, a, b, shift * (fw/32)));
    } else if (((shift % 2) == 0) && (fw < 32)) {
        return fwCast(fw, mvmd_dslli(2 * fw, a, b, shift / 2));
    }
    const unsigned fieldCount = mBitBlockWidth/fw;
    if ((fw == 32) || (hasFeature(Feature::AVX512_BW) && (fw == 16)))   {
        //llvm::errs() << " fw = " << fw << ", shift = " << shift << "\n";
        Type * fwTy = getIntNTy(fw);
        SmallVector<Constant *, 16> indices(fieldCount);
        for (unsigned i = 0; i < fieldCount; i++) {
            indices[i] = ConstantInt::get(fwTy, i + fieldCount - shift);
        }
        return mvmd_shuffle2(fw, fwCast(fw, b), fwCast(fw, a), ConstantVector::get(indices));
    } else {
        unsigned field32_shift = (shift * fw) / 32;
        unsigned bit_shift = (shift * fw) % 32;
        Value * const L = simd_slli(32, mvmd_dslli(32, a, b, field32_shift), bit_shift);
        Value * const R = simd_srli(32, mvmd_dslli(32, a, b, field32_shift + 1), 32-bit_shift);
        assert (L->getType() == R->getType());
        return fwCast(fw, CreateOr(L, R));
    }
}

Value * IDISA_AVX512F_Builder::simd_popcount(unsigned fw, Value * a) {
     if (fw == 512) {
         Constant * zero16xi8 = Constant::getNullValue(FixedVectorType::get(getInt8Ty(), 16));
         Constant * zeroInt32 = Constant::getNullValue(getInt32Ty());
         Value * c = simd_popcount(64, a);
         //  Should probably use _mm512_reduce_add_epi64, but not found in LLVM 3.8
         Function * pack64_8_func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_pmov_qb_512);
         // popcounts of 64 bit fields will always fit in 8 bit fields.
         // We don't need the masked version of this, but the unmasked intrinsic was not found.
         c = CreateCall(pack64_8_func->getFunctionType(), pack64_8_func, {c, zero16xi8, Constant::getAllOnesValue(getInt8Ty())});
         Function * horizSADfunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_sse2_psad_bw);
         c = CreateCall(horizSADfunc->getFunctionType(), horizSADfunc, {c, zero16xi8});
         return CreateInsertElement(allZeroes(), CreateExtractElement(c, zeroInt32), zeroInt32);
    }
    if (hasFeature(Feature::AVX512_VPOPCNTDQ) && (fw == 32 || fw == 64)){
        //llvm should use vpopcntd or vpopcntq instructions
        return CreatePopcount(fwCast(fw, a));
    }
    if (hasFeature(Feature::AVX512_BW) && (fw == 64)) {
        Function * horizSADfunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_psad_bw_512);
        return CreateCall(horizSADfunc->getFunctionType(), horizSADfunc, {fwCast(8, simd_popcount(8, a)), fwCast(8, allZeroes())});
    }
    //https://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation
    if((fw == 64) && (mBitBlockWidth == 512)){
        Constant * m1Arr[8];
        Constant * m1;
        for (unsigned int i = 0; i < 8; i++) {
            m1Arr[i] = getInt64(0x5555555555555555);
        }
        m1 = ConstantVector::get({m1Arr, 8});

        Constant * m2Arr[8];
        Constant * m2;
        for (unsigned int i = 0; i < 8; i++) {
            m2Arr[i] = getInt64(0x3333333333333333);
        }
        m2 = ConstantVector::get({m2Arr, 8});

        Constant * m4Arr[8];
        Constant * m4;
        for (unsigned int i = 0; i < 8; i++) {
            m4Arr[i] = getInt64(0x0f0f0f0f0f0f0f0f);
        }
        m4 = ConstantVector::get({m4Arr, 8});

        Constant * h01Arr[8];
        Constant * h01;
        for (unsigned int i = 0; i < 8; i++) {
            h01Arr[i] = getInt64(0x0101010101010101);
        }
        h01 = ConstantVector::get({h01Arr, 8});

        a = simd_sub(fw, a, simd_and(simd_srli(fw, a, 1), m1));
        a = simd_add(fw, simd_and(a, m2), simd_and(simd_srli(fw, a, 2), m2));
        a = simd_and(simd_add(fw, a, simd_srli(fw, a, 4)), m4);
        return simd_srli(fw, simd_mult(fw, a, h01), 56);

    }
    return IDISA_Builder::simd_popcount(fw, a);
}

Value * IDISA_AVX512F_Builder::hsimd_signmask(unsigned fw, Value * a) {
    //IDISA_Builder::hsimd_signmask outperforms IDISA_AVX2_Builder::hsimd_signmask
    //when run with BlockSize=512
    return IDISA_Builder::hsimd_signmask(fw, a);
}

Value * IDISA_AVX512F_Builder::esimd_mergeh(unsigned fw, Value * a, Value * b) {
    if (hasFeature(Feature::AVX512_BW) && ((fw == 1) || (fw == 2))) {
        // Bit interleave using shuffle.
        // Make a shuffle table that translates the lower 4 bits of each byte in
        // order to spread out the bits: xxxxdcba => .d.c.b.a
        // We use two copies of the table for the AVX2 _mm256_shuffle_epi8
        Constant * interleave_table = bit_interleave_byteshuffle_table(fw);
        // Merge the bytes.
        Value * byte_merge = esimd_mergeh(8, a, b);
        Function * shufFn = Intrinsic::getDeclaration(getModule(),  Intrinsic::x86_avx512_pshuf_b_512);
        Value * low_bits = CreateCall(shufFn->getFunctionType(), shufFn, {interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8)))});
        Value * high_bits = simd_slli(16, CreateCall(shufFn->getFunctionType(), shufFn, {interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))}), fw);
        Value * lo_move_back = simd_srli(16, low_bits, 8-fw);
        Value * hi_move_fwd = simd_slli(16, high_bits, 8-fw);
        return simd_or(simd_if(1, simd_himask(16), high_bits, low_bits), simd_or(lo_move_back, hi_move_fwd));
    }
    if (fw == 8)   {
        const unsigned fieldCount = mBitBlockWidth/fw;
        SmallVector<Constant *, 8> Idxs(fieldCount/2);
        for (unsigned i = 0; i < fieldCount / 2; i++) {
            Idxs[i] = getInt32(i+fieldCount/2); // selects elements from first reg.
        }
        Constant * high_indexes = ConstantVector::get(Idxs);
        Value * a_high = CreateShuffleVector(fwCast(8, a), UndefValue::get(fwVectorType(8)), high_indexes);
        Value * b_high = CreateShuffleVector(fwCast(8, b), UndefValue::get(fwVectorType(8)), high_indexes);
        Value * a_ext = CreateZExt(a_high, fwVectorType(16));
        Value * b_ext = CreateZExt(b_high, fwVectorType(16));
        Value * rslt = simd_or(a_ext, simd_slli(16, b_ext, 8));
        return rslt;
    }
    // Otherwise use default AVX2 logic.
    return IDISA_AVX2_Builder::esimd_mergeh(fw, a, b);
}

Value * IDISA_AVX512F_Builder::esimd_mergel(unsigned fw, Value * a, Value * b) {
    if (hasFeature(Feature::AVX512_BW) && ((fw == 1) || (fw == 2))) {
        // Bit interleave using shuffle.
        // Make a shuffle table that translates the lower 4 bits of each byte in
        // order to spread out the bits: xxxxdcba => .d.c.b.a
        // We use two copies of the table for the AVX2 _mm256_shuffle_epi8
        Constant * interleave_table = bit_interleave_byteshuffle_table(fw);
        // Merge the bytes.
        Value * byte_merge = esimd_mergel(8, a, b);
        Function * shufFn = Intrinsic::getDeclaration(getModule(),  Intrinsic::x86_avx512_pshuf_b_512);
        Value * low_bits = CreateCall(shufFn->getFunctionType(), shufFn, {interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8)))});
        Value * high_bits = simd_slli(16, CreateCall(shufFn->getFunctionType(), shufFn, {interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))}), fw);
        Value * lo_move_back = simd_srli(16, low_bits, 8-fw);
        Value * hi_move_fwd = simd_slli(16, high_bits, 8-fw);
        return simd_or(simd_if(1, simd_himask(16), high_bits, low_bits), simd_or(lo_move_back, hi_move_fwd));
    }
    if (fw == 8) {
        const unsigned fieldCount = mBitBlockWidth/fw;
        SmallVector<Constant *, 8> Idxs(fieldCount/2);
        for (unsigned i = 0; i < fieldCount / 2; i++) {
            Idxs[i] = getInt32(i); // selects elements from first reg.
        }
        Constant * low_indexes = ConstantVector::get(Idxs);
        Value * a_low = CreateShuffleVector(fwCast(8, a), UndefValue::get(fwVectorType(8)), low_indexes);
        Value * b_low = CreateShuffleVector(fwCast(8, b), UndefValue::get(fwVectorType(8)), low_indexes);
        Value * a_ext = CreateZExt(a_low, fwVectorType(16));
        Value * b_ext = CreateZExt(b_low, fwVectorType(16));
        Value * rslt = simd_or(a_ext, simd_slli(16, b_ext, 8));
        return rslt;
    }
    // Otherwise use default AVX2 logic.
    return IDISA_AVX2_Builder::esimd_mergel(fw, a, b);
}

Value * IDISA_AVX512F_Builder::simd_if(unsigned fw, Value * cond, Value * a, Value * b) {
    if (fw == 1) {
        // Form the 8-bit table for simd-if based on the bitwise values from cond, a and b.
        //   (cond, a, b) =  (111), (110), (101), (100), (011), (010), (001), (000)
        // if(cond, a, b) =    1      1      0      0      1      0      1      0    = 0xCA
        return simd_ternary(0xCA, bitCast(cond), bitCast(a), bitCast(b));
    }
    return IDISA_AVX2_Builder::simd_if(fw, cond, a, b);
}

Value * IDISA_AVX512F_Builder::simd_ternary(unsigned char mask, Value * a, Value * b, Value * c) {
    assert (a->getType() == b->getType());
    assert (b->getType() == c->getType());

    if (mask == 0) {
        return allZeroes();
    }
    if (mask == 0xFF) {
        return allOnes();
    }

    unsigned char not_a_mask = mask & 0x0F;
    unsigned char a_mask = (mask >> 4) & 0x0F;
    if (a_mask == not_a_mask) {
        return simd_binary(a_mask, b, c);
    }

    unsigned char b_mask = ((mask & 0xC0) >> 4) | ((mask & 0x0C) >> 2);
    unsigned char not_b_mask = ((mask & 0x30) >> 2) | (mask & 0x03);
    if (b_mask == not_b_mask) {
        return simd_binary(b_mask, a, c);
    }

    unsigned char c_mask = ((mask & 0x80) >> 4) | ((mask & 0x20) >> 3) | ((mask & 0x08) >> 2) | ((mask & 02) >> 1);
    unsigned char not_c_mask = ((mask & 0x40) >> 3) | ((mask & 0x10) >> 2) | ((mask & 0x04) >> 1) | (mask & 01);
    if (c_mask == not_c_mask) {
        return simd_binary(c_mask, a, b);
    }

    Constant * simd_mask = ConstantInt::get(getInt32Ty(), mask);
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(7, 0, 0)
    Function * ternLogicFn = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_pternlog_d_512);
    Constant * writemask = ConstantInt::getAllOnesValue(getInt16Ty());
    Value * args[5] = {fwCast(32, a), fwCast(32, b), fwCast(32, c), simd_mask, writemask};
#else
    Function * ternLogicFn = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_pternlog_d_512);
    Value * args[4] = {fwCast(32, a), fwCast(32, b), fwCast(32, c), simd_mask};
#endif
    return bitCast(CreateCall(ternLogicFn->getFunctionType(), ternLogicFn, args));
}

std::pair<Value *, Value *> IDISA_AVX512F_Builder::bitblock_advance(Value * a, Value * shiftin, unsigned shift) {
    assert (a->getType() == mBitBlockType);
    if (shift == 1 && shiftin->getType() == getInt8Ty()) {
        const uint32_t fw = 64;
        Value * const ci_mask = CreateBitCast(shiftin, FixedVectorType::get(getInt1Ty(), 8));
        Value * const v8xi64_1 = simd_fill(fw, ConstantInt::get(getInt64Ty(), 0x8000000000000000));
        Value * const ecarry_in = CreateSelect(ci_mask, v8xi64_1, Constant::getNullValue(FixedVectorType::get(getInt64Ty(), 8)));
        Value * const a1 = mvmd_dslli(fw, a, ecarry_in, shift);
        Value * const result = simd_or(CreateLShr(a1, fw - shift), CreateShl(fwCast(fw, a), shift));

        std::vector<Constant *> v(8, ConstantInt::get(getInt64Ty(), (uint64_t) -1));
        v[7] = ConstantInt::get(getInt64Ty(), 0x7fffffffffffffff);
        Value * const v8xi64_cout_mask = ConstantVector::get(ArrayRef<Constant *>(v));
        Value * shiftout = CreateICmpUGT(a, v8xi64_cout_mask);
        shiftout = CreateBitCast(shiftout, getInt8Ty());
        return std::make_pair(shiftout, result);
    } else {
        return IDISA_AVX2_Builder::bitblock_advance(a, shiftin, shift);
    }
}

IDISA_AVX_Builder::IDISA_AVX_Builder(LLVMContext & C, const FeatureSet & featureSet, unsigned vectorWidth, unsigned laneWidth)
: IDISA_Builder(C, featureSet, AVX_width, vectorWidth, laneWidth)
, IDISA_SSE2_Builder(C, featureSet, vectorWidth, laneWidth) {

}

IDISA_AVX2_Builder::IDISA_AVX2_Builder(LLVMContext & C, const FeatureSet & featureSet, unsigned vectorWidth, unsigned laneWidth)
: IDISA_Builder(C, featureSet, AVX_width, vectorWidth, laneWidth)
, IDISA_AVX_Builder(C, featureSet, vectorWidth, laneWidth) {

}

IDISA_AVX512F_Builder::IDISA_AVX512F_Builder(LLVMContext & C, const FeatureSet & featureSet, unsigned vectorWidth, unsigned laneWidth)
: IDISA_Builder(C, featureSet, AVX512_width, vectorWidth, laneWidth)
, IDISA_AVX2_Builder(C, featureSet, vectorWidth, laneWidth) {

}

}
