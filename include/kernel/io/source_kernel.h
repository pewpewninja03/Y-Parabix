#pragma once

#include <kernel/core/kernel.h>
namespace kernel { class KernelBuilder; }

namespace kernel {

/* The MMapSourceKernel is a simple wrapper for an external MMap file buffer.
   The doSegment method of this kernel feeds one segment at a time to a
   pipeline. */

class MMapSourceKernel final : public SegmentOrientedKernel {
    friend class FDSourceKernel;
public:
    MMapSourceKernel(LLVMTypeSystemInterface & ts, Scalar * const fd, StreamSet * const outputStream);
    void linkExternalMethods(KernelBuilder & b) override;
    void generateInitializeMethod(KernelBuilder & b) override {
        generateInitializeMethod(b, mCodeUnitWidth);
    }
    void generateDoSegmentMethod(KernelBuilder & b) override {
        generateDoSegmentMethod(b, mCodeUnitWidth);
    }
    void generateFinalizeMethod(KernelBuilder & b) override {
        freeBuffer(b, mCodeUnitWidth);
    }
    llvm::Value * generateExpectedOutputSizeMethod(KernelBuilder & b) override {
        return generateExpectedOutputSizeMethod(b, mCodeUnitWidth);
    }
private:
    static void generatLinkExternalFunctions(KernelBuilder & b);
    static void generateInitializeMethod(KernelBuilder & b, const unsigned codeUnitWidth);
    static void generateDoSegmentMethod(KernelBuilder & b, const unsigned codeUnitWidth);
    static llvm::Value * generateExpectedOutputSizeMethod(KernelBuilder & b, const unsigned codeUnitWidth);
    static void freeBuffer(KernelBuilder & b, const unsigned codeUnitWidth);
protected:
    const unsigned mCodeUnitWidth;
};

class ReadSourceKernel final : public SegmentOrientedKernel {
    friend class FDSourceKernel;
public:
    ReadSourceKernel(LLVMTypeSystemInterface & ts, Scalar * const fd, StreamSet * const outputStream);
    void linkExternalMethods(KernelBuilder & b) override;
    void generateDoSegmentMethod(KernelBuilder & b) override {
        generateDoSegmentMethod(b, mCodeUnitWidth);
    }
    llvm::Value * generateExpectedOutputSizeMethod(KernelBuilder & b) override {
        return generateExpectedOutputSizeMethod(b, mCodeUnitWidth);
    }
protected:
    static void generatLinkExternalFunctions(KernelBuilder & b);
    static void generateDoSegmentMethod(KernelBuilder & b, const unsigned codeUnitWidth);
    static llvm::Value * generateExpectedOutputSizeMethod(KernelBuilder & b, const unsigned codeUnitWidth);
private:
    const unsigned mCodeUnitWidth;
};

class FDSourceKernel final : public SegmentOrientedKernel {
public:
    FDSourceKernel(LLVMTypeSystemInterface & ts, Scalar * const useMMap, Scalar * const fd, StreamSet * const outputStream);
    void linkExternalMethods(KernelBuilder & b) override;
    void generateInitializeMethod(KernelBuilder & b) override;
    void generateDoSegmentMethod(KernelBuilder & b) override;
    void generateFinalizeMethod(KernelBuilder & b) override;
    llvm::Value * generateExpectedOutputSizeMethod(KernelBuilder &) override;
protected:
    const unsigned mCodeUnitWidth;
};

class MemorySourceKernel final : public SegmentOrientedKernel {
public:
    MemorySourceKernel(LLVMTypeSystemInterface & ts, Scalar * fileSource, Scalar * fileItems, StreamSet * const outputStream);
protected:
    void generateInitializeMethod(KernelBuilder & b) override;
    void generateDoSegmentMethod(KernelBuilder & b) override;
    llvm::Value * generateExpectedOutputSizeMethod(KernelBuilder &) override;
};

}

