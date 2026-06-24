#include <idisa/idisa_arm_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsAArch64.h>
#include <llvm/IR/Module.h>
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(17, 0, 0)
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif

using namespace llvm;

namespace IDISA {

std::string IDISA_ARM_Builder::getBuilderUniqueName() { return mBitBlockWidth != 128 ? "ARM_" + std::to_string(mBitBlockWidth) : "ARM";}

Value* IDISA_ARM_Builder::simd_popcount(unsigned fw, Value * a) {
    if (getVectorBitWidth(a) != ARM_width || fw < 8 || fw % 8 != 0) {
        return IDISA_Builder::simd_popcount(fw, a);
    }

    // There is a CNT instruction offered by NEON that counts set bits in each byte.
    // It only exists for vectors of i8, i.e. <8 x i8> and <16 x i8>. For some reason,
    // LLVM exposes an instrinsic for this instruction but fails to select it
    // during compilation. As a workaround we use the LLVM ctpop instrinsic which does
    // the right thing and emits CNT.
    Value* countInBytes = CreatePopcount(fwCast(8, a));

    if (fw == 8) { // if `a` is a vector of i8 then we're already done
        return countInBytes;
    } else if (fw == 128) {
        auto addv = Intrinsic::getDeclaration(getModule(),
                                              Intrinsic::aarch64_neon_uaddv,
                                              { getInt32Ty(), FixedVectorType::get(getInt8Ty(), 16) });

        auto popcnt = CreateCall(addv->getFunctionType(),
                                     addv,
                                     fwCast(8, countInBytes));

        // It appears that when fw == 128 most calling code expects we return 2xi64
        return CreateInsertElement(fwCast(64, allZeroes()),
                                   CreateZExt(popcnt, getInt64Ty()),
                                   Constant::getNullValue(getInt32Ty()));
    } else {
        // addParirsW: pairwise widening add
        // Adds each pair of fields in a vector together and stores
        // the result in a vector whose fields are twice as wide as
        // the source vector
        auto addPairsW = [this](unsigned _fw, Value* _a) -> Value* {
            unsigned nElems = getVectorBitWidth(_a) / _fw;
            unsigned destFw = _fw * 2;
            unsigned destNElems = nElems / 2;

            auto low = CreateExtractVector(FixedVectorType::get(getIntNTy(_fw), nElems / 2),
                                           _a,
                                           ConstantInt::get(getInt64Ty(), 0));
            auto hi = CreateExtractVector(FixedVectorType::get(getIntNTy(_fw), nElems / 2),
                                           _a,
                                           ConstantInt::get(getInt64Ty(), nElems / 2));

            auto lowExt = CreateZExt(low, FixedVectorType::get(getIntNTy(destFw), destNElems));
            auto hiExt = CreateZExt(hi, FixedVectorType::get(getIntNTy(destFw), destNElems));

            auto addp = Intrinsic::getDeclaration(getModule(),
                                                  Intrinsic::aarch64_neon_addp,
                                                  FixedVectorType::get(getIntNTy(destFw), destNElems));
            return fwCast(destFw, CreateCall(addp->getFunctionType(), addp, {lowExt, hiExt}));
        };

        // Add pairs together and widen each field until we have reduced
        // to the destination field width
        Value* result = countInBytes;
        for (unsigned thisFw = 8; thisFw < fw; thisFw <<= 1) {
            result = addPairsW(thisFw, result);
        }
        return result;
    }
}

Value * IDISA_ARM_Builder::simd_bitreverse(unsigned fw, Value * a) {

    if (fw < 8 || getVectorBitWidth(a) != ARM_width) {
        return IDISA_Builder::simd_bitreverse(fw, a);
    }

    // First reverse the bits in each byte
    auto rbit = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_sve_rbit, fwVectorType(fw));
    if (fw == 8) {
        return CreateCall(rbit->getFunctionType(), rbit, fwCast(8, a));
    }
    Function* refBytesInFields = nullptr;

    // Then reverse the bytes in each field
    if (fw == 64) {
        refBytesInFields = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_sve_revw);
    } else if (fw == 32) {
        refBytesInFields = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_sve_revh);
    } else if (fw == 16) {
        refBytesInFields = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_sve_revb);
    } else {
        return IDISA_Builder::simd_bitreverse(fw, a);
    }

    auto bitsInBytesRevsd = CreateCall(rbit->getFunctionType(), rbit, fwCast(8, a));
    return CreateCall(refBytesInFields->getFunctionType(), refBytesInFields, fwCast(fw, bitsInBytesRevsd));
}

Value * IDISA_ARM_Builder::mvmd_shuffle(unsigned fw, Value * data_table, Value * index_vector) {
  if (mBitBlockWidth == 128 && fw > 8) {
    // Create a table for shuffling with smaller field widths.
    const unsigned fieldCount = mBitBlockWidth/fw;
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
    Value * rslt = mvmd_shuffle(half_fw, data_table, half_fw_indexes);
    return rslt;
  }
  if (mBitBlockWidth == 128 && fw == 8) {
    Function * shuf8Func = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_neon_tbl1, FixedVectorType::get(getInt8Ty(), 16));
    return fwCast(8, CreateCall(shuf8Func->getFunctionType(), shuf8Func, {fwCast(8, data_table), fwCast(8, simd_select_lo(fw, index_vector))}));
  }
  return IDISA_Builder::mvmd_shuffle(fw, data_table, index_vector);
}

Value * IDISA_ARM_Builder::mvmd_shuffle2(unsigned fw, Value * table0, Value * table1, Value * index_vector) {
    if (mBitBlockWidth == 128 && fw == 8) {
        Function * shuf8Func = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_neon_tbl2, FixedVectorType::get(getInt8Ty(), 16));
        Value * rslt = CreateCall(shuf8Func->getFunctionType(), shuf8Func, {fwCast(8, table0), fwCast(8, table1), fwCast(8, index_vector)});
            return rslt;
    }
    return IDISA_Builder::mvmd_shuffle2(fw, table0, table1, index_vector);
}

Value * IDISA_ARM_Builder::hsimd_packl(unsigned fw, Value * a, Value * b) {
    if ((fw >= 16) && (fw <= 64) && (getVectorBitWidth(a) == ARM_width)) {
        int nElems = getVectorBitWidth(a) / fw;
        int halfFw = fw / 2;
        Function* uzp1_fn = Intrinsic::getDeclaration(getModule(),
                                                      Intrinsic::aarch64_sve_uzp1,
                                                      FixedVectorType::get(getIntNTy(halfFw), nElems * 2));
        return CreateCall(uzp1_fn->getFunctionType(), uzp1_fn, {fwCast(halfFw, a), fwCast(halfFw, b)});
    }
    // Otherwise use default logic.
    return IDISA_Builder::hsimd_packl(fw, a, b);
}

Value * IDISA_ARM_Builder::hsimd_packh(unsigned fw, Value * a, Value * b) {
    if ((fw >= 16) && (fw <= 64) && (getVectorBitWidth(a) == ARM_width)) {
        int nElems = getVectorBitWidth(a) / fw;
        int halfFw = fw / 2;
        Function* uzp2_fn = Intrinsic::getDeclaration(getModule(),
                                                      Intrinsic::aarch64_sve_uzp2,
                                                      FixedVectorType::get(getIntNTy(halfFw), nElems * 2));
        return CreateCall(uzp2_fn->getFunctionType(), uzp2_fn, {fwCast(halfFw, a), fwCast(halfFw, b)});
    }
    // Otherwise use default logic.
    return IDISA_Builder::hsimd_packh(fw, a, b);
}

Value * IDISA_ARM_Builder::hsimd_packus(unsigned fw, Value * a, Value * b) {
  if ((fw == 16) && (getVectorBitWidth(a) == ARM_width)) {
    Function * vqmovun_s16_func = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_neon_uqxtn, FixedVectorType::get(getInt8Ty(), 8));
    Value * sat_a = CreateCall(vqmovun_s16_func->getFunctionType(), vqmovun_s16_func, fwCast(16, a));
    Value * sat_b = CreateCall(vqmovun_s16_func->getFunctionType(), vqmovun_s16_func, fwCast(16, b));
    return fwCast(8, CreateDoubleVector(sat_a, sat_b));
  }
  // Otherwise use default logic.
  return IDISA_Builder::hsimd_packus(fw, a, b);
}

Value * IDISA_ARM_Builder::esimd_mergeh(unsigned fw, Value * a, Value * b) {

  if ((fw >= 16) && (fw <= 64) && (getVectorBitWidth(a) == ARM_width)) {
    int nElms = getVectorBitWidth(a) / fw;
    int halfFw = fw / 2;
    Function * zip2_fn = Intrinsic::getDeclaration(getModule(),
                                                 Intrinsic::aarch64_sve_zip2,
                                                 FixedVectorType::get(getIntNTy(halfFw), nElms * 2));
    return CreateCall(zip2_fn->getFunctionType(), zip2_fn, {fwCast(halfFw, a), fwCast(halfFw, b)});
  }
  return IDISA_Builder::esimd_mergeh(fw, a, b);
}

Value * IDISA_ARM_Builder::esimd_mergel(unsigned fw, Value * a, Value * b) {
  if ((fw >= 16) && (fw <= 64) && (getVectorBitWidth(a) == ARM_width)) {
    int nElms = getVectorBitWidth(a) / fw;
    int halfFw = fw / 2;
    Function * zip1_fn = Intrinsic::getDeclaration(getModule(),
                                                 Intrinsic::aarch64_sve_zip1,
                                                 FixedVectorType::get(getIntNTy(halfFw), nElms * 2));
    return CreateCall(zip1_fn->getFunctionType(), zip1_fn, {fwCast(halfFw, a), fwCast(halfFw, b)});
  }
  return IDISA_Builder::esimd_mergel(fw, a, b);
}

}
