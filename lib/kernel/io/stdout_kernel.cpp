/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/io/stdout_kernel.h>

#include <llvm/IR/Module.h>
#include <kernel/core/kernel_builder.h>
#include <toolchain/toolchain.h>
#include <kernel/core/streamset.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>

namespace llvm { class Type; }

using namespace llvm;

namespace kernel {

void StdOutKernel::linkExternalMethods(KernelBuilder & b) {
    b.LinkFunction("write", write);
}

void StdOutKernel::generateDoSegmentMethod(KernelBuilder & b) {

    Module * const m = b.getModule();

    Value * bytesDone = b.getProcessedItemCount("codeUnitBuffer");
    Value * codeUnitBuffer = b.getRawInputPointer("codeUnitBuffer", bytesDone);
    Value * bytesToDo = b.getAccessibleItemCount("codeUnitBuffer");
    if (mCodeUnitWidth != 8) {
        codeUnitBuffer = b.CreatePointerCast(codeUnitBuffer, b.getInt8PtrTy());
        if (LLVM_UNLIKELY(mCodeUnitWidth > 8)) {
            bytesToDo = b.CreateMul(bytesToDo, b.getSize(mCodeUnitWidth / 8));
        } else if (LLVM_UNLIKELY(mCodeUnitWidth < 8)) {
            bytesToDo = b.CreateUDiv(bytesToDo, b.getSize(8 / mCodeUnitWidth));
        }
    }
    FixedArray<Value *, 3> args3;
    args3[0] = b.getInt32(STDOUT_FILENO);
    args3[1] = codeUnitBuffer;
    args3[2] = bytesToDo;
    Function * writeFunc = m->getFunction("write");
    b.CreateCall(writeFunc, args3);
}

StdOutKernel::StdOutKernel(LLVMTypeSystemInterface & ts, StreamSet *codeUnitBuffer)
: SegmentOrientedKernel(ts, "stdout" + std::to_string(codeUnitBuffer->getFieldWidth()),
// input
{Binding{"codeUnitBuffer", codeUnitBuffer}}
// output & scalars
, {}, {}, {}, {})
, mCodeUnitWidth(codeUnitBuffer->getFieldWidth()) {
    setStride((8 * BUFSIZ) / mCodeUnitWidth);
    addAttribute(SideEffecting());
}

void FileSink::linkExternalMethods(KernelBuilder & b) {
    b.LinkFunction("write", write);
}

void FileSink::generateInitializeMethod(KernelBuilder & b) {
    BasicBlock * const nonNullFileName = b.CreateBasicBlock("nonNullFileName");
    BasicBlock * const nonEmptyFileName = b.CreateBasicBlock("nonEmptyFileName");
    BasicBlock * const setTerminationOnFailure = b.CreateBasicBlock("setTerminationOnFailure");
    BasicBlock * const fileSinkInitExit = b.CreateBasicBlock("fileSinkInitExit");
    Value * const fileName = b.getScalarField("fileName");

    BasicBlock * const entryBlock = b.GetInsertBlock();
    b.CreateLikelyCondBr(b.CreateIsNotNull(fileName), nonNullFileName, fileSinkInitExit);

    b.SetInsertPoint(nonNullFileName);
    Value * const fileNameLength = b.CreateStrlenCall(fileName);
    b.CreateLikelyCondBr(b.CreateIsNotNull(fileNameLength), nonEmptyFileName, fileSinkInitExit);

    b.SetInsertPoint(nonEmptyFileName);
    // Make a temporary file name template with the characters "XXXXXX" appended as required by mkstemp.
    Constant * suffixPlusNullLength = b.getSize(7);
    Value * const temporaryFileName = b.CreatePointerCast(b.CreateMalloc(b.CreateAdd(fileNameLength, suffixPlusNullLength)), b.getInt8PtrTy());
    b.CreateMemCpy(temporaryFileName, fileName, fileNameLength, 1);
    #ifdef BACKUP_OLDFILE
    b.CreateMemCpy(b.CreateGEP0(tmpFileNamePtr, fileNameLength), b.GetString(".saved"), suffixPlusNullLength, 1);
    b.CreateRenameCall(fileName, tmpFileNamePtr);
    #else
    b.CreateUnlinkCall(fileName);
    #endif
    b.CreateMemCpy(b.CreateGEP(b.getInt8Ty(), temporaryFileName, fileNameLength), b.GetString("XXXXXX"), suffixPlusNullLength, 1);
    Value * const temporaryFd = b.CreateMkstempCall(temporaryFileName);
    ConstantInt * const errorCodeFd = b.getInt32(-1);
    Value * failure = b.CreateICmpEQ(temporaryFd, errorCodeFd);
    b.CreateUnlikelyCondBr(failure, setTerminationOnFailure, fileSinkInitExit);

    b.SetInsertPoint(setTerminationOnFailure);
    b.setTerminationSignal();
    b.CreateBr(fileSinkInitExit);

    b.SetInsertPoint(fileSinkInitExit);

    Constant * const nullPointer = ConstantPointerNull::get(b.getInt8PtrTy());
    PHINode * const temporaryFileNamePhi = b.CreatePHI(b.getInt8PtrTy(), 3);
    temporaryFileNamePhi->addIncoming(nullPointer, entryBlock);
    temporaryFileNamePhi->addIncoming(nullPointer, nonNullFileName);
    temporaryFileNamePhi->addIncoming(temporaryFileName, nonEmptyFileName);
    temporaryFileNamePhi->addIncoming(nullPointer, setTerminationOnFailure);

    ConstantInt * const stdOutFd = b.getInt32(STDOUT_FILENO);
    PHINode * const fileDescriptorPhi = b.CreatePHI(b.getInt32Ty(), 3);
    fileDescriptorPhi->addIncoming(stdOutFd, entryBlock);
    fileDescriptorPhi->addIncoming(stdOutFd, nonNullFileName);
    fileDescriptorPhi->addIncoming(temporaryFd, nonEmptyFileName);
    fileDescriptorPhi->addIncoming(errorCodeFd, setTerminationOnFailure);

    b.setScalarField("temporaryFileName", temporaryFileNamePhi);
    b.setScalarField("fileDescriptor", fileDescriptorPhi);
}

void FileSink::generateDoSegmentMethod(KernelBuilder & b) {
    Module * const m = b.getModule();

    Value * bytesDone = b.getProcessedItemCount("codeUnitBuffer");
    Value * codeUnitBuffer = b.getRawInputPointer("codeUnitBuffer", bytesDone);
    Value * bytesToDo = b.getAccessibleItemCount("codeUnitBuffer");
    if (mCodeUnitWidth != 8) {
        codeUnitBuffer = b.CreatePointerCast(codeUnitBuffer, b.getInt8PtrTy());
        if (LLVM_UNLIKELY(mCodeUnitWidth > 8)) {
            bytesToDo = b.CreateMul(bytesToDo, b.getSize(mCodeUnitWidth / 8));
        } else if (LLVM_UNLIKELY(mCodeUnitWidth < 8)) {
            bytesToDo = b.CreateUDiv(bytesToDo, b.getSize(8 / mCodeUnitWidth));
        }
    }

    Value * const fileDescriptor = b.getScalarField("fileDescriptor");

    FixedArray<Value *, 3> args3;
    args3[0] = fileDescriptor;
    args3[1] = codeUnitBuffer;
    args3[2] = bytesToDo;
    Function * writeFunc = m->getFunction("write");
    b.CreateCall(writeFunc, args3);
}

void FileSink::generateFinalizeMethod(KernelBuilder & b) {
    BasicBlock * const hasTemporaryFile = b.CreateBasicBlock("hasTemporaryFile");
    BasicBlock * const exit = b.CreateBasicBlock("exit");
    Value * const fileDescriptor = b.getScalarField("fileDescriptor");
    Value * const temporaryFileName = b.getScalarField("temporaryFileName");
    b.CreateLikelyCondBr(b.CreateIsNotNull(temporaryFileName), hasTemporaryFile, exit);

    b.SetInsertPoint(hasTemporaryFile);
    b.CreateCloseCall(fileDescriptor);
    Value * const fileName = b.getScalarField("fileName");
    b.CreateRenameCall(temporaryFileName, fileName);
    b.CreateFree(temporaryFileName);
    b.CreateBr(exit);

    b.SetInsertPoint(exit);
}

FileSink::FileSink(LLVMTypeSystemInterface & ts, Scalar * outputFileName, StreamSet * codeUnitBuffer)
: SegmentOrientedKernel(ts, "filesink" + std::to_string(codeUnitBuffer->getFieldWidth()),
// input
{Binding{"codeUnitBuffer", codeUnitBuffer}},
// output
{},
// input scalars
{Binding{"fileName", outputFileName}},
// output scalars
{},
// internal scalars
{InternalScalar{ts.getInt8PtrTy(), "temporaryFileName"},
 InternalScalar{ts.getInt32Ty(), "fileDescriptor"}})
, mCodeUnitWidth(codeUnitBuffer->getFieldWidth()) {
    setStride((8 * BUFSIZ) / mCodeUnitWidth);
    addAttribute(SideEffecting());
}

}
