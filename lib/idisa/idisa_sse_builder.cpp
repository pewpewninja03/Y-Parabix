/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_sse_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(10, 0, 0)
#include <llvm/IR/IntrinsicsX86.h>
#endif

using namespace llvm;

namespace IDISA {

std::string IDISA_SSE_Builder::getBuilderUniqueName() { return mBitBlockWidth != 128 ? "SSE_" + std::to_string(mBitBlockWidth) : "SSE";}
std::string IDISA_SSE2_Builder::getBuilderUniqueName() { return mBitBlockWidth != 128 ? "SSE2_" + std::to_string(mBitBlockWidth) : "SSE2";}
std::string IDISA_SSSE3_Builder::getBuilderUniqueName() { return mBitBlockWidth != 128 ? "SSSE3_" + std::to_string(mBitBlockWidth) : "SSSE3";}

Value * IDISA_SSE_Builder::hsimd_signmask(const unsigned fw, Value * a) {
    // Produces wrong result on AVX2 with fw = 16
    // const unsigned SSE_blocks = getVectorBitWidth(a)/SSE_width;
    // if (SSE_blocks > 1) {
    //     Value * a_lo = CreateHalfVectorLow(a);
    //     Value * a_hi = CreateHalfVectorHigh(a);
    //     if ((fw == 8 * SSE_blocks) || (fw >= 32 * SSE_blocks)) {
    //         return IDISA_SSE_Builder::hsimd_signmask(fw/2, IDISA_SSE_Builder::hsimd_packh(fw, a_hi, a_lo));
    //     }
    //     unsigned maskWidth = getVectorBitWidth(a)/fw;
    //     Type * maskTy = getIntNTy(maskWidth);
    //     Value * mask_lo = CreateZExtOrTrunc(hsimd_signmask(fw, a_lo), maskTy);
    //     Value * mask_hi = CreateZExtOrTrunc(hsimd_signmask(fw, a_hi), maskTy);
    //     return CreateOr(CreateShl(mask_hi, maskWidth/2), mask_lo);
    // }
    // SSE special cases using Intrinsic::x86_sse_movmsk_ps (fw=32 only)
    if (fw == 32) {
        Function * signmask_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_sse_movmsk_ps);
        Type * bitBlock_f32type = FixedVectorType::get(getFloatTy(), mBitBlockWidth/32);
        Value * a_as_ps = CreateBitCast(a, bitBlock_f32type);
        if (getVectorBitWidth(a) == SSE_width) {
            return CreateCall(signmask_f32func->getFunctionType(), signmask_f32func, a_as_ps);
        }
    }
    // Otherwise use default logic.
    return IDISA_Builder::hsimd_signmask(fw, a);
}

Value * IDISA_SSE_Builder::mvmd_compress(unsigned fw, Value * a, Value * selector) {
    if (LLVM_LIKELY(mBitBlockWidth == 128)) {
        if (fw == 64) {
            Constant * keep[2] = {ConstantInt::get(getInt64Ty(), 1), ConstantInt::get(getInt64Ty(), 3)};
            Constant * keep_mask = ConstantVector::get({keep, 2});
            Constant * shift[2] = {ConstantInt::get(getInt64Ty(), 2), ConstantInt::get(getInt64Ty(), 0)};
            Constant * shifted_mask = ConstantVector::get({shift, 2});
            Value * a_srli1 = mvmd_srli(64, a, 1);
            Value * bdcst = simd_fill(64, CreateZExt(selector, getInt64Ty()));
            Value * kept = simd_and(simd_eq(64, simd_and(keep_mask, bdcst), keep_mask), a);
            Value * shifted = simd_and(a_srli1, simd_eq(64, shifted_mask, bdcst));
            return simd_or(kept, shifted);
        }
        else if (fw == 32) {
            Value * bdcst = simd_fill(32, CreateZExtOrTrunc(selector, getInt32Ty()));
            Constant * fieldBit[4] =
            {ConstantInt::get(getInt32Ty(), 1), ConstantInt::get(getInt32Ty(), 2),
                ConstantInt::get(getInt32Ty(), 4), ConstantInt::get(getInt32Ty(), 8)};
            Constant * fieldMask = ConstantVector::get({fieldBit, 4});
            Value * a_selected = simd_and(simd_eq(32, fieldMask, simd_and(fieldMask, bdcst)), a);
            Constant * rotateInwards[4] =
            {ConstantInt::get(getInt32Ty(), 1), ConstantInt::get(getInt32Ty(), 0),
                ConstantInt::get(getInt32Ty(), 3), ConstantInt::get(getInt32Ty(), 2)};
            Constant * rotateVector = ConstantVector::get({rotateInwards, 4});
            Value * rotated = CreateShuffleVector(fwCast(32, a_selected), UndefValue::get(fwVectorType(fw)), rotateVector);
            Constant * rotate_bit[2] = {ConstantInt::get(getInt64Ty(), 2), ConstantInt::get(getInt64Ty(), 4)};
            Constant * rotate_mask = ConstantVector::get({rotate_bit, 2});
            Value * rotateControl = simd_eq(64, fwCast(64, simd_and(bdcst, rotate_mask)), allZeroes());
            Value * centralResult = simd_if(1, rotateControl, rotated, a_selected);
            Value * delete_marks_lo = CreateAnd(CreateZExtOrTrunc(CreateNot(selector), getInt32Ty()), getInt32(3));
            Value * delCount_lo = CreateSub(delete_marks_lo, CreateLShr(delete_marks_lo, 1));
            return mvmd_srl(32, centralResult, delCount_lo);
        }
    }
    return IDISA_Builder::mvmd_compress(fw, a, selector);
}


Value * IDISA_SSE2_Builder::hsimd_packh(unsigned fw, Value * a, Value * b) {
    if ((fw == 16) && (getVectorBitWidth(a) == SSE_width)) {
        Function * packuswb_func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_sse2_packuswb_128);
        return CreateCall(packuswb_func->getFunctionType(), packuswb_func, {simd_srli(16, a, 8), simd_srli(16, b, 8)});
    }
    // Otherwise use default logic.
    return IDISA_SSE_Builder::hsimd_packh(fw, a, b);
}

Value * IDISA_SSE2_Builder::hsimd_packl(unsigned fw, Value * a, Value * b) {
    if ((fw == 16) && (getVectorBitWidth(a) == SSE_width)) {
        Value * mask = simd_lomask(16);
        return hsimd_packus(fw, fwCast(16, simd_and(a, mask)), fwCast(16, simd_and(b, mask)));
    }
    // Otherwise use default logic.
    return IDISA_SSE_Builder::hsimd_packl(fw, a, b);
}

Value * IDISA_SSE2_Builder::hsimd_packus(unsigned fw, Value * a, Value * b) {
    if ((fw == 16) && (getVectorBitWidth(a) == SSE_width)) {
        Function * packuswb_func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_sse2_packuswb_128);
        return CreateCall(packuswb_func->getFunctionType(), packuswb_func, {fwCast(16, a), fwCast(16, b)});
    }
    // Otherwise use default logic.
    return IDISA_SSE_Builder::hsimd_packus(fw, a, b);
}

Value * IDISA_SSE2_Builder::hsimd_signmask(unsigned fw, Value * a) {
    // SSE2 special case using Intrinsic::x86_sse2_movmsk_pd (fw=32 only)
    if (getVectorBitWidth(a) == SSE_width) {
        if (fw == 64) {
            Function * signmask_f64func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_sse2_movmsk_pd);
            Type * bitBlock_f64type = FixedVectorType::get(getDoubleTy(), mBitBlockWidth/64);
            Value * a_as_pd = CreateBitCast(a, bitBlock_f64type);
            return CreateCall(signmask_f64func->getFunctionType(), signmask_f64func, a_as_pd);
        }
        if (fw == 8) {
            Function * pmovmskb_func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_sse2_pmovmskb_128);
            return CreateCall(pmovmskb_func->getFunctionType(), pmovmskb_func, fwCast(8, a));
        }
    }
    // Otherwise use default SSE logic.
    return IDISA_SSE_Builder::hsimd_signmask(fw, a);
}

#if 0

#define SHIFT_FIELDWIDTH 64
//#define LEAVE_CARRY_UNNORMALIZED

// full shift producing {shiftout, shifted}
std::pair<Value *, Value *> IDISA_SSE2_Builder::bitblock_advance(Value * const a, Value * shiftin, const unsigned shift) {
    Value * shifted = nullptr;
    Value * shiftout = nullptr;
    Type * shiftTy = shiftin->getType();
    assert (a->getType() == mBitBlockType);
    assert (mBitBlockType->getElementType()->getIntegerBitWidth() == SHIFT_FIELDWIDTH);
    if (LLVM_UNLIKELY(shift == 0)) {
        return std::pair<Value *, Value *>(Constant::getNullValue(shiftTy), a);
    }
    if (shiftTy != mBitBlockType) {
        assert (shiftTy->isIntegerTy());
        if (LLVM_LIKELY(shiftTy->getIntegerBitWidth() < SHIFT_FIELDWIDTH)) {
            shiftin = CreateZExt(shiftin, getIntNTy(SHIFT_FIELDWIDTH));
        }
        shiftin = CreateInsertElement(ConstantVector::getNullValue(a->getType()), shiftin, getInt32(0));
    }
    assert (shiftin->getType() == mBitBlockType);

    auto getShiftout = [&](Value * v) {
        if (v->getType() != shiftTy) {
            v = CreateExtractElement(v, getInt32(0));
            if (LLVM_LIKELY(shiftTy->getIntegerBitWidth() < SHIFT_FIELDWIDTH)) {
                v = CreateTrunc(v, shiftTy);
            }
        }
        return v;
    };

    if (LLVM_UNLIKELY(shift == mBitBlockWidth)) {
        return std::pair<Value *, Value *>(getShiftout(a), shiftin);
    }
#ifndef LEAVE_CARRY_UNNORMALIZED
    if (LLVM_UNLIKELY((shift % 8) == 0)) { // Use a single whole-byte shift, if possible.
        shifted = CreateOr(bitCast(mvmd_slli(8, a, shift / 8)), shiftin);
        shiftout = bitCast(mvmd_srli(8, a, (mBitBlockWidth - shift) / 8));
    } else {
        Value * shiftback = simd_srli(SHIFT_FIELDWIDTH, a, SHIFT_FIELDWIDTH - (shift % SHIFT_FIELDWIDTH));
        Value * shiftfwd = simd_slli(SHIFT_FIELDWIDTH, a, shift % SHIFT_FIELDWIDTH);
        if (LLVM_LIKELY(shift < SHIFT_FIELDWIDTH)) {
            shiftout = mvmd_srli(SHIFT_FIELDWIDTH, shiftback, mBitBlockWidth/SHIFT_FIELDWIDTH - 1);
            shifted = CreateOr(CreateOr(shiftfwd, shiftin), mvmd_slli(SHIFT_FIELDWIDTH, shiftback, 1));
        } else {
            shiftout = CreateOr(shiftback, mvmd_srli(SHIFT_FIELDWIDTH, shiftfwd, 1));
            shifted = CreateOr(shiftin, mvmd_slli(SHIFT_FIELDWIDTH, shiftfwd, (mBitBlockWidth - shift) / SHIFT_FIELDWIDTH));
            if ((shift + SHIFT_FIELDWIDTH) < mBitBlockWidth) {
                shiftout = mvmd_srli(SHIFT_FIELDWIDTH, shiftout, (mBitBlockWidth - shift) / SHIFT_FIELDWIDTH);
                shifted = CreateOr(shifted, mvmd_slli(SHIFT_FIELDWIDTH, shiftback, shift/SHIFT_FIELDWIDTH + 1));
            }
        }
    }
#else
    shiftout = a;
    if (LLVM_UNLIKELY((shift % 8) == 0)) { // Use a single whole-byte shift, if possible.
        shifted = mvmd_dslli(8, a, shiftin, (mBitBlockWidth - shift) / 8);
    }
    else if (LLVM_LIKELY(shift < SHIFT_FIELDWIDTH)) {
        Value * ahead = mvmd_dslli(SHIFT_FIELDWIDTH, a, shiftin, mBitBlockWidth / SHIFT_FIELDWIDTH - 1);
        shifted = simd_or(simd_srli(SHIFT_FIELDWIDTH, ahead, SHIFT_FIELDWIDTH - shift), simd_slli(SHIFT_FIELDWIDTH, a, shift));
    }
    else {
        throw std::runtime_error("Unsupported shift.");
    }
#endif
    assert (shifted->getType() == mBitBlockType);
    assert (shiftout->getType() == mBitBlockType);
    return std::pair<Value *, Value *>(getShiftout(shiftout), shifted);
}

#endif

Value * IDISA_SSE2_Builder::mvmd_shuffle(unsigned fw, Value * a, Value * index_vector) {
    if ((getVectorBitWidth(a) == SSE_width) && (fw == 64)) {
        // First create a vector with exchanged values of the 2 fields.
        Constant * idx[2] = {ConstantInt::get(getInt32Ty(), 1), ConstantInt::get(getInt32Ty(), 0)};
        Value * exchanged = CreateShuffleVector(a, UndefValue::get(fwVectorType(fw)), ConstantVector::get({idx, 2}));
        // bits that change if the value in a needs to be exchanged.
        Value * changed = simd_xor(a, exchanged);
        // Now create a mask to select between original and exchanged values.
        Constant * xchg[2] = {ConstantInt::get(getInt64Ty(), 1), ConstantInt::get(getInt64Ty(), 0)};
        Value * xchg_vec = ConstantVector::get({xchg, 2});
        Constant * oneSplat = getSplat(2, ConstantInt::get(getInt64Ty(), 1));
        Value * exchange_mask = simd_eq(fw, simd_and(index_vector, oneSplat), xchg_vec);
        Value * rslt = simd_xor(simd_and(changed, exchange_mask), a);
        return rslt;
    }
    return IDISA_SSE_Builder::mvmd_shuffle(fw, a, index_vector);
}
    
std::vector<Value *> IDISA_SSE2_Builder::simd_pext(unsigned fw, std::vector<Value *> v, Value * extract_mask) {
    if ((getVectorBitWidth(v[0]) == SSE_width) && (fw > 8)) {
        std::vector<Value *> w = v;
        w.push_back(extract_mask); // Compress the masks as well.
        w = simd_pext(fw/2, w, extract_mask);
        Value * compressed_masks = simd_select_lo(fw, w.back());
        Value * multiplier = simd_add(fw, compressed_masks, simd_fill(fw, getIntN(fw, 1)));
        std::vector<Value *> c(v.size());
        for (unsigned i = 0; i < v.size(); i++) {
            c[i] = simd_or(simd_mult(fw, multiplier, simd_srli(fw, w[i], fw/2)), simd_select_lo(fw, w[i]));
        }
        return c;
    }
    return IDISA_SSE_Builder::simd_pext(fw, v, extract_mask);
}

Value * IDISA_SSSE3_Builder::esimd_mergeh(unsigned fw, Value * a, Value * b) {
    if ((getVectorBitWidth(a) == SSE_width) && ((fw == 1) || (fw == 2))) {
        Constant * interleave_table = bit_interleave_byteshuffle_table(fw);
        // Merge the bytes.
        Value * byte_merge = esimd_mergeh(8, a, b);
        Value * low_bits = mvmd_shuffle(8, interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8))));
        Value * high_bits = simd_slli(16, mvmd_shuffle(8, interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))), fw);
        // For each 16-bit field, interleave the low bits of the two bytes.
        low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8-fw));
        // For each 16-bit field, interleave the high bits of the two bytes.
        high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8-fw));
        return simd_or(low_bits, high_bits);
    }
    // Otherwise use default SSE logic.
    return IDISA_SSE2_Builder::esimd_mergeh(fw, a, b);
}

Value * IDISA_SSSE3_Builder::esimd_mergel(unsigned fw, Value * a, Value * b) {
    if ((getVectorBitWidth(a) == SSE_width) && ((fw == 1) || (fw == 2))) {
        Constant * interleave_table = bit_interleave_byteshuffle_table(fw);
        // Merge the bytes.
        Value * byte_merge = esimd_mergel(8, a, b);
        Value * low_bits = mvmd_shuffle(8, interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8))));
        Value * high_bits = simd_slli(16, mvmd_shuffle(8, interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))), fw);
        // For each 16-bit field, interleave the low bits of the two bytes.
        low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8-fw));
        // For each 16-bit field, interleave the high bits of the two bytes.
        high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8-fw));
        return simd_or(low_bits, high_bits);
    }
    // Otherwise use default SSE2 logic.
    return IDISA_SSE2_Builder::esimd_mergel(fw, a, b);
}

Value * IDISA_SSSE3_Builder::mvmd_shuffle(unsigned fw, Value * a, Value * index_vector) {
    if (getVectorBitWidth(a) == SSE_width) {
        if (fw > 8) {
            // Create a table for shuffling with smaller field widths.
            const unsigned fieldCount = mBitBlockWidth/fw;
#if 0
            Constant * idxMask = getSplat(fieldCount, ConstantInt::get(getIntNTy(fw), fieldCount-1));
            Value * idx = simd_and(index_vector, idxMask);
            unsigned half_fw = fw/2;
            unsigned field_count = mBitBlockWidth/half_fw;
            // Build a ConstantVector of alternating 0 and 1 values.
            SmallVector<Constant *, 16> Idxs(field_count);
            for (unsigned int i = 0; i < field_count; i++) {
                Idxs[i] = ConstantInt::get(getIntNTy(fw/2), i & 1);
            }
            Constant * splat01 = ConstantVector::get(Idxs);

            Value * half_fw_indexes = simd_or(idx, mvmd_slli(half_fw, idx, 1));
            half_fw_indexes = simd_add(fw, simd_add(fw, half_fw_indexes, half_fw_indexes), splat01);
            Value * rslt = mvmd_shuffle(half_fw, a, half_fw_indexes);
            return rslt;
#endif
            ConstantInt * multiplier = 0;
            ConstantInt * addition = 0;
            if (fw == 64) {
                multiplier = getInt64(0x0808080808080808ULL);
                addition =   getInt64(0x0706050403020100ULL);
            } else if (fw == 32) {
                multiplier = getInt32(0x04040404);
                addition =   getInt32(0x03020100);
            } else if (fw == 16) {
                multiplier = getInt16(0x0202);
                addition =   getInt16(0x0100);
            }
            Value * A = CreateMul(fwCast(fw, index_vector), getSplat(fieldCount, multiplier));
            index_vector = CreateOr(A, getSplat(fieldCount, addition));
            return fwCast(fw, mvmd_shuffle(8, a, index_vector));
        } else if (fw == 8) {
            Function * shuf8Func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_ssse3_pshuf_b_128);
            return CreateCall(shuf8Func->getFunctionType(), shuf8Func, {fwCast(8, a), fwCast(8, simd_and(index_vector, simd_lomask(8)))});
        }
    }
    return IDISA_SSE2_Builder::mvmd_shuffle(fw, a, index_vector);
}

Value * IDISA_SSSE3_Builder::mvmd_compress(unsigned fw, Value * a, Value * select_mask) {
    //if (LLVM_LIKELY(mBitBlockWidth == 128)) {
        // simd_pext

    //}
    return IDISA_SSE2_Builder::mvmd_compress(fw, a, select_mask);
}

Value * IDISA_SSSE3_Builder::mvmd_expand(unsigned fw, Value * a, Value * select_mask)  {
    //if (LLVM_LIKELY(mBitBlockWidth == 128)) {
        // simd_pdep
    //}
    return IDISA_SSE2_Builder::mvmd_expand(fw, a, select_mask);
}

}
