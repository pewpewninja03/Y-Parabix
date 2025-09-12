#pragma once

#include <idisa/idisa_builder.h>

namespace IDISA {

constexpr unsigned ARM_width = 128;

class IDISA_ARM_Builder : public virtual IDISA_Builder {
public:
    static constexpr unsigned NativeBitBlockWidth = ARM_width;

    IDISA_ARM_Builder(llvm::LLVMContext & C, const FeatureSet & featureSet, unsigned bitBlockWidth, unsigned laneWidth)
    : IDISA_Builder(C, featureSet, ARM_width, bitBlockWidth, laneWidth) {

    }

    virtual std::string getBuilderUniqueName() override;
    llvm::Value* simd_popcount(unsigned fw, llvm::Value* a) override;
    llvm::Value* simd_bitreverse(unsigned fw, llvm::Value* a) override;
    llvm::Value * esimd_mergeh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * esimd_mergel(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packl(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packus(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector) override;
    llvm::Value * mvmd_shuffle2(unsigned fw, llvm::Value * table0, llvm::Value * table1, llvm::Value * index_vector) override;

    ~IDISA_ARM_Builder() {}
};

}

