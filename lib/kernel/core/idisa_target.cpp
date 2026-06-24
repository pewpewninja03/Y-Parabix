/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/core/idisa_target.h>

#include <toolchain/toolchain.h>
#include <idisa/idisa_i64_builder.h>
#ifdef PARABIX_ARM_TARGET
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(16, 0, 0)
#include <llvm/TargetParser/AArch64TargetParser.h>
#endif
#include <idisa/idisa_arm_builder.h>
#endif
#ifdef PARABIX_X86_TARGET
#include <idisa/idisa_sse_builder.h>
#include <idisa/idisa_avx_builder.h>
#endif
#ifdef PARABIX_NVPTX_TARGET
#include <idisa/idisa_nvptx_builder.h>
#endif
#include <llvm/IR/Module.h>

#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(16, 0, 0)
#include <llvm/TargetParser/Triple.h>
#else
#include <llvm/ADT/Triple.h>
#endif

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/core/kernel_builder.h>

#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(17, 0, 0)
#include <llvm/TargetParser/Host.h>
#elif LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(11, 0, 0)
#include <llvm/Support/Host.h>
#endif

#define ADD_IF_FOUND(Flag, Value) if (features.lookup(Value)) featureSet.set((size_t)Feature::Flag)

using namespace kernel;
using namespace llvm;

struct Features {
    bool hasAVX;
    bool hasAVX2;
    bool hasAVX512F;
    Features() : hasAVX(0), hasAVX2(0), hasAVX512F(0) { }
};

Features getHostCPUFeatures(const StringMap<bool> & features) {
    Features hostCPUFeatures;
    hostCPUFeatures.hasAVX = features.lookup("avx");
    hostCPUFeatures.hasAVX2 = features.lookup("avx2");
    hostCPUFeatures.hasAVX512F = features.lookup("avx512f");
    return hostCPUFeatures;
}

bool ARM_available() {
#ifdef PARABIX_ARM_TARGET
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(16, 0, 0)
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(17, 0, 0)
    auto info = llvm::AArch64::parseCpu(sys::getHostCPUName());
    std::vector<StringRef> extNames;
    if (info) {
        llvm::AArch64::getExtensionFeatures(info->Arch.DefaultExts | info->DefaultExtensions, extNames);
    }
#else
    const llvm::AArch64::CpuInfo & info = llvm::AArch64::parseCpu(sys::getHostCPUName());
    std::vector<StringRef> extNames;
    llvm::AArch64::getExtensionFeatures(info.Arch.DefaultExts | info.DefaultExtensions, extNames);
#endif
    for (const auto eName : extNames) {
        //llvm::errs() << "Extension: " << eName << "\n";
        if (eName == "+neon") return true;
    }
    return false;
#else
    StringMap<bool> features;
    if (LLVM_UNLIKELY(!sys::getHostCPUFeatures(features))) {
        return false;
    }
    return features.lookup("neon");
#endif
#endif
    return false;
}

bool AVX2_available() {
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(19, 0, 0)
    StringMap<bool> features;
    if (LLVM_UNLIKELY(!sys::getHostCPUFeatures(features))) {
        return false;
    }
    #else
    const auto features = sys::getHostCPUFeatures();
    #endif
    return features.lookup("avx2");
}

bool AVX512BW_available() {
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(19, 0, 0)
    StringMap<bool> features;
    if (LLVM_UNLIKELY(!sys::getHostCPUFeatures(features))) {
        return false;
    }
    #else
    const auto features = sys::getHostCPUFeatures();
    #endif
    return features.lookup("avx512bw");
}

namespace IDISA {

KernelBuilder * GetIDISA_Builder(llvm::LLVMContext & C, const StringMap<bool> & features) {
    IDISA_Builder::FeatureSet featureSet;
    if (codegen::BlockSize == 64) {
        return new KernelBuilderImpl<IDISA_I64_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }
#ifdef PARABIX_ARM_TARGET
    if (LLVM_LIKELY(codegen::BlockSize == 0)) {  // No BlockSize override: use processor SIMD width
        codegen::BlockSize = 128;
    }
    if (ARM_available()) {
        return new KernelBuilderImpl<IDISA_ARM_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }
#endif
#ifdef PARABIX_X86_TARGET

    const auto HasAVX = features.lookup("avx");
    const auto HasAVX2 = features.lookup("avx2");
    const auto HasAVX512F = features.lookup("avx512f");

    if (LLVM_LIKELY(codegen::BlockSize == 0)) {  // No BlockSize override: use processor SIMD width
        if (LLVM_UNLIKELY(HasAVX512F)) {
            codegen::BlockSize = 512;
        } else if (HasAVX2) {
            codegen::BlockSize = 256;
        } else {
            codegen::BlockSize = 128;
        }
    } else if (((codegen::BlockSize & (codegen::BlockSize - 1)) != 0) || (codegen::BlockSize < 64)) {
        llvm::report_fatal_error("BlockSize must be a power of 2 and >=64");
    }

    if (HasAVX || HasAVX2) {
        ADD_IF_FOUND(AVX_BMI, "bmi");
        ADD_IF_FOUND(AVX_BMI2, "bmi2");
    }
    if (HasAVX512F) {
        ADD_IF_FOUND(AVX512_CD, "avx512cd");
        ADD_IF_FOUND(AVX512_BW, "avx512bw");
        ADD_IF_FOUND(AVX512_DQ, "avx512dq");
        ADD_IF_FOUND(AVX512_VL, "avx512vl");
        // AVX512_VBMI, AVX512_VBMI2 and AVX512_VPOPCNTDQ  have not been tested as we
        //did not have hardware support. It should work in theory (tm)
        ADD_IF_FOUND(AVX512_VBMI, "avx512vbmi");
        ADD_IF_FOUND(AVX512_VBMI2, "avx512vbmi2");
        ADD_IF_FOUND(AVX512_VPOPCNTDQ, "avx512vpopcntdq");
    }
    // AVX512BW builder can only be used for BlockSize multiples of 512
    if (codegen::BlockSize >= 512 && HasAVX512F) {
        return new KernelBuilderImpl<IDISA_AVX512F_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }
    if (codegen::BlockSize >= 256) {
        // AVX2 or AVX builders can only be used for BlockSize multiples of 256
        if (HasAVX2) {
            return new KernelBuilderImpl<IDISA_AVX2_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
        } else if (HasAVX) {
            return new KernelBuilderImpl<IDISA_AVX_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
        }
    }
    if (codegen::BlockSize == 128) {
        if (features.lookup("ssse3")) {
            return new KernelBuilderImpl<IDISA_SSSE3_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
        } else {
            return new KernelBuilderImpl<IDISA_SSE2_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
        }
    }
#endif
    llvm::errs() << "BlockSize 64 default!\n";
    codegen::BlockSize = 64;
    return new KernelBuilderImpl<IDISA_I64_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
}
#ifdef PARABIX_NVPTX_TARGET
KernelBuilder * GetIDISA_GPU_Builder(llvm::LLVMContext & C) {
    return new KernelBuilderImpl<IDISA_NVPTX20_Builder>(C, 64 * 64, 64);
}
#endif
}
