/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/io/source_kernel.h>

#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <llvm/IR/Module.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <toolchain/toolchain.h>
#include <boost/interprocess/mapped_region.hpp>
#include <llvm/Support/raw_ostream.h>
#include <unistd.h>
#include <sys/mman.h>

using namespace llvm;

inline unsigned getPageSize() {
    return boost::interprocess::mapped_region::get_page_size();
}

extern "C" uint64_t file_size(const uint32_t fd) {
    struct stat st;
    if (LLVM_UNLIKELY(fstat(fd, &st) != 0)) {
        st.st_size = 0;
    }
    return st.st_size;
}

namespace kernel {

/// MMAP SOURCE KERNEL

void MMapSourceKernel::generatLinkExternalFunctions(KernelBuilder & b) {
    b.LinkFunction("file_size", file_size);
    b.LinkFunction("mmap", mmap);
    b.LinkFunction("madvise", madvise);
    b.LinkFunction("munmap", munmap);
}

void MMapSourceKernel::generateInitializeMethod(KernelBuilder & b, const unsigned codeUnitWidth, const unsigned stride) {

    BasicBlock * const emptyFile = b.CreateBasicBlock("emptyFile");
    BasicBlock * const nonEmptyFile = b.CreateBasicBlock("NonEmptyFile");
    BasicBlock * const exit = b.CreateBasicBlock("Exit");
    IntegerType * const sizeTy = b.getSizeTy();
    Value * const fd = b.getScalarField("fileDescriptor");
    PointerType * const codeUnitPtrTy = b.getIntNTy(codeUnitWidth)->getPointerTo();
    Function * const fileSizeFn = b.getModule()->getFunction("file_size"); assert (fileSizeFn);
    FunctionType * fTy = fileSizeFn->getFunctionType();
    Value * fileSize = b.CreateZExtOrTrunc(b.CreateCall(fTy, fileSizeFn, fd), sizeTy);
    b.CreateLikelyCondBr(b.CreateIsNotNull(fileSize), nonEmptyFile, emptyFile);

    b.SetInsertPoint(nonEmptyFile);
    Value * const fileBuffer = b.CreatePointerCast(b.CreateFileSourceMMap(fd, fileSize), codeUnitPtrTy);
    b.setScalarField("buffer", fileBuffer);
    b.setBaseAddress("sourceBuffer", fileBuffer);
    Value * fileItems = fileSize;
    if (LLVM_UNLIKELY(codeUnitWidth > 8)) {
        fileItems = b.CreateUDiv(fileSize, b.getSize(codeUnitWidth / 8));
    }
    b.setScalarField("fileItems", fileItems);
    b.setCapacity("sourceBuffer", fileItems);
    b.CreateBr(exit);

    b.SetInsertPoint(emptyFile);
    ConstantInt * const STRIDE_BYTES = b.getSize(stride * codeUnitWidth);
    Value * const emptyFilePtr = b.CreatePointerCast(b.CreateAnonymousMMap(STRIDE_BYTES), codeUnitPtrTy);
    b.setScalarField("buffer", emptyFilePtr);
    b.setBaseAddress("sourceBuffer", emptyFilePtr);
    b.setScalarField("fileItems", STRIDE_BYTES);
    b.setTerminationSignal();
    b.CreateBr(exit);

    b.SetInsertPoint(exit);
}


void MMapSourceKernel::generateDoSegmentMethod(KernelBuilder & b, const unsigned codeUnitWidth, const unsigned stride) {

    BasicBlock * const dropPages = b.CreateBasicBlock("dropPages");
    BasicBlock * const checkRemaining = b.CreateBasicBlock("checkRemaining");
    BasicBlock * const setTermination = b.CreateBasicBlock("setTermination");
    BasicBlock * const exit = b.CreateBasicBlock("mmapSourceExit");

    Value * const numOfStrides = b.getNumOfStrides();
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b.CreateAssert(b.CreateIsNotNull(numOfStrides),
                        "Internal error: %s.numOfStrides cannot be 0", b.GetString("MMapSource"));
    }

    // TODO: could we improve overall performance by trying to "preload" the data by reading it? This would increase
    // the cost of this kernel but might allow the first kernel to read the file data be better balanced with it.

    const auto pageSize = getPageSize();
    Value * const desiredItems = b.CreateMul(numOfStrides, b.getSize(stride));
    ConstantInt * const CODE_UNIT_BYTES = b.getSize(codeUnitWidth / 8);

    Value * const consumedItems = b.getConsumedItemCount("sourceBuffer");
    Value * const consumedBytes = b.CreateMul(consumedItems, CODE_UNIT_BYTES);
    Value * const consumedPageOffset = b.CreateRoundDownRational(consumedBytes, pageSize);
    Value * const consumedBuffer = b.getRawOutputPointer("sourceBuffer", consumedPageOffset);
    Value * const readableBuffer = b.getScalarField("buffer");
    Value * const unnecessaryBytes = b.CreateSub(consumedPageOffset, b.CreatePtrToInt(readableBuffer, b.getSizeTy()));

    Module * const m = b.getModule();
    Function * MAdviseFunc = m->getFunction("madvise");
    assert (MAdviseFunc);
    FixedArray<Value *, 3> args;

    // avoid calling madvise unless an actual page table change could occur
    b.CreateLikelyCondBr(b.CreateIsNotNull(unnecessaryBytes), dropPages, checkRemaining);
    b.SetInsertPoint(dropPages);
    // instruct the OS that it can safely drop any fully consumed pages
    #ifndef __APPLE__
    args[0] = readableBuffer;
    args[1] = consumedPageOffset;
    args[2] = b.getInt32(MADV_DONTNEED);
    b.CreateCall(MAdviseFunc, args);
    #endif
    b.CreateBr(checkRemaining);

    // determine whether or not we've exhausted the "safe" region of the file buffer
    b.SetInsertPoint(checkRemaining);
    Value * const producedItems = b.getProducedItemCount("sourceBuffer");
    Value * const nextProducedItems = b.CreateAdd(producedItems, desiredItems);
    Value * const fileItems = b.getScalarField("fileItems");
    Value * const lastPage = b.CreateICmpULE(fileItems, nextProducedItems);
    b.CreateUnlikelyCondBr(lastPage, setTermination, exit);

    // If this is the last page, create a temporary buffer of up to two pages size, copy the unconsumed data
    // and zero any bytes that are not used.
    b.SetInsertPoint(setTermination);
    b.setTerminationSignal();
    b.setProducedItemCount("sourceBuffer", fileItems);
    b.CreateBr(exit);

    b.SetInsertPoint(exit);
    PHINode * producedPhi = b.CreatePHI(b.getSizeTy(), 2);
    producedPhi->addIncoming(nextProducedItems, checkRemaining);
    producedPhi->addIncoming(fileItems, setTermination);
    Value * const producedBytes = b.CreateMul(producedPhi, CODE_UNIT_BYTES);
    Value * const length = b.CreateSub(producedBytes, consumedPageOffset);

    args[0] = b.CreatePointerCast(consumedBuffer, cast<PointerType>(MAdviseFunc->getArg(0)->getType()));
    args[1] = length;
    args[2] = b.getInt32(MADV_WILLNEED);
    b.CreateCall(MAdviseFunc, args);

}
void MMapSourceKernel::freeBuffer(KernelBuilder & b, const unsigned codeUnitWidth) {
    Value * const fileItems = b.getScalarField("fileItems");
    Constant * const CODE_UNIT_BYTES = b.getSize(codeUnitWidth / 8);
    Value * const fileSize = b.CreateMul(fileItems, CODE_UNIT_BYTES);
    Module * const m = b.getModule();
    Function * MUnmapFunc = m->getFunction("munmap");
    assert (MUnmapFunc);
    FixedArray<Value *, 2> args;
    args[0] = b.CreatePointerCast(b.getBaseAddress("sourceBuffer"), b.getVoidPtrTy());
    args[1] = fileSize;
    b.CreateCall(MUnmapFunc, args);
}

Value * MMapSourceKernel::generateExpectedOutputSizeMethod(KernelBuilder & b, const unsigned codeUnitWidth) {
    return b.getScalarField("fileItems");
}

void MMapSourceKernel::linkExternalMethods(KernelBuilder & b) {
    MMapSourceKernel::generatLinkExternalFunctions(b);
}

/// READ SOURCE KERNEL

void ReadSourceKernel::generatLinkExternalFunctions(KernelBuilder & b) {
    b.LinkFunction("read", read);
    b.LinkFunction("file_size", file_size);
}

template <typename IntTy>
inline IntTy round_up_to(const IntTy x, const IntTy y) {
    assert(is_power_2(y));
    return (x + y - 1) & -y;
}

void ReadSourceKernel::generateInitializeMethod(KernelBuilder & b, const unsigned codeUnitWidth, const unsigned stride) {

}

void ReadSourceKernel::generateDoSegmentMethod(KernelBuilder & b, const unsigned codeUnitWidth, const unsigned stride) {

    Value * const numOfStrides = b.getNumOfStrides();
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b.CreateAssert(b.CreateIsNotNull(numOfStrides),
                        "Internal error: %s.numOfStrides cannot be 0", b.GetString("ReadSource"));
    }

    BasicBlock * const readData = b.CreateBasicBlock("ReadData");
    BasicBlock * const readIncomplete = b.CreateBasicBlock("readIncomplete");
    BasicBlock * const setTermination = b.CreateBasicBlock("SetTermination");
    BasicBlock * const readExit = b.CreateBasicBlock("ReadExit");

    Value * const segmentItems = b.CreateMul(numOfStrides, b.getSize(stride));
    ConstantInt * const codeUnitBytes = b.getSize(codeUnitWidth / 8);
    IntegerType * const sizeTy = b.getSizeTy();

    // Can we append to our existing buffer without impacting any subsequent kernel?
    b.reserveCapacity("sourceBuffer", segmentItems);

    Value * const segmentBytes = b.CreateMul(segmentItems, codeUnitBytes);
    Value * const fd = b.getScalarField("fileDescriptor");
    Value * const produced = b.getProducedItemCount("sourceBuffer");
    BasicBlock * const entryBlock = b.GetInsertBlock();
    b.CreateBr(readData);

    // Regardless of whether we're simply appending data or had to allocate a new buffer, read a new page
    // of data into the input source buffer. This may involve multiple read calls.
    b.SetInsertPoint(readData);
    PHINode * const bytesToRead = b.CreatePHI(sizeTy, 2);
    bytesToRead->addIncoming(segmentBytes, entryBlock);
    PHINode * const producedSoFar = b.CreatePHI(sizeTy, 2);
    producedSoFar->addIncoming(produced, entryBlock);

    Value * const sourceBuffer = b.CreatePointerCast(b.getRawOutputPointer("sourceBuffer", producedSoFar), b.getInt8PtrTy());

    Function *  const preadFunc = b.getModule()->getFunction("read");
    FixedArray<Value *, 3> args;
    args[0] = fd;
    args[1] = sourceBuffer;
    args[2] = bytesToRead;

    Value * const bytesRead = b.CreateCall(preadFunc, args);

    // There are 4 possibile results from read:
    // bytesRead == -1: an error occurred
    // bytesRead == 0: EOF, no bytes read
    // 0 < bytesRead < bytesToRead:  some data read (more may be available)
    // bytesRead == bytesToRead, the full amount requested was read.
    b.CreateUnlikelyCondBr(b.CreateICmpNE(bytesToRead, bytesRead), readIncomplete, readExit);

    b.SetInsertPoint(readIncomplete);
    // Keep reading until a the full stride is read, or there is no more data.
    Value * moreToRead = b.CreateSub(bytesToRead, bytesRead);
    Value * readSoFar = b.CreateSub(segmentBytes, moreToRead);
    Value * const itemsRead = b.CreateUDiv(readSoFar, codeUnitBytes);
    Value * const itemsBuffered = b.CreateAdd(produced, itemsRead);
    bytesToRead->addIncoming(moreToRead, readIncomplete);
    producedSoFar->addIncoming(itemsBuffered, readIncomplete);
    b.CreateCondBr(b.CreateICmpSGT(bytesRead, b.getSize(0)), readData, setTermination);

    // ... set the termination signal.
    b.SetInsertPoint(setTermination);
    Value * const itemsPending = b.CreateAdd(produced, segmentItems);
    Value * const bytesToZero = b.CreateMul(b.CreateSub(itemsPending, itemsBuffered), codeUnitBytes);
    b.CreateMemZero(b.getRawOutputPointer("sourceBuffer", itemsBuffered), bytesToZero);
    b.setScalarField("fileItems", itemsBuffered);
    b.setTerminationSignal();
    b.setProducedItemCount("sourceBuffer", itemsBuffered);
    b.CreateBr(readExit);

    b.SetInsertPoint(readExit);
}

Value * ReadSourceKernel::generateExpectedOutputSizeMethod(KernelBuilder & b, const unsigned codeUnitWidth) {
    Value * const fd = b.getScalarField("fileDescriptor");
    Function * const fileSizeFn = b.getModule()->getFunction("file_size"); assert (fileSizeFn);
    FunctionType * fTy = fileSizeFn->getFunctionType();
    return b.CreateZExtOrTrunc(b.CreateCall(fTy, fileSizeFn, fd), b.getSizeTy());
}

void ReadSourceKernel::freeBuffer(KernelBuilder & b, const unsigned codeUnitWidth) {
//    Module * m = b.getModule();
//    ConstantInt * const codeUnitBytes = b.getSize(codeUnitWidth / 8);
//    Value * const buffer = b.getScalarField("buffer");
//    Value * const capacity = b.getScalarField("effectiveCapacity");
//    Function * destroyBuffer = m->getFunction(__DESTROY_CIRCULAR_BUFFER);
//    FixedArray<Value *, 3> destroyArgs;
//    destroyArgs[0] = b.CreatePointerCast(buffer, b.getInt8PtrTy());
//    destroyArgs[1] = b.CreateMul(capacity, codeUnitBytes);
//    destroyArgs[2] = b.getSize(0);
//    b.CreateCall(destroyBuffer, destroyArgs);
}

void ReadSourceKernel::finalizeThreadLocalMethod(KernelBuilder & b, const unsigned codeUnitWidth) {
//    Module * m = b.getModule();
//    ConstantInt * const codeUnitBytes = b.getSize(codeUnitWidth / 8);
//    Function * destroyBuffer = m->getFunction(__DESTROY_CIRCULAR_BUFFER);
//    FixedArray<Value *, 3> destroyArgs;
//    Value * const priorBuffer = b.getScalarField("ancillaryBuffer");
//    Value * const priorCapacity = b.getScalarField("ancillaryCapacity");
//    destroyArgs[0] = b.CreatePointerCast(priorBuffer, b.getInt8PtrTy());
//    destroyArgs[1] = b.CreateMul(priorCapacity, codeUnitBytes);
//    destroyArgs[2] = b.getSize(0);
//    b.CreateCall(destroyBuffer, destroyArgs);
}

void ReadSourceKernel::linkExternalMethods(KernelBuilder & b) {
    ReadSourceKernel::generatLinkExternalFunctions(b);
}

/// Hybrid MMap/Read source kernel

void FDSourceKernel::generateInitializeMethod(KernelBuilder & b) {
    BasicBlock * initializeRead = b.CreateBasicBlock("initializeRead");
    BasicBlock * checkFileSize = b.CreateBasicBlock("checkFileSize");
    BasicBlock * initializeMMap = b.CreateBasicBlock("initializeMMap");
    BasicBlock * initializeDone = b.CreateBasicBlock("initializeDone");

    // The source will use MMapSource or readSoure kernel logic depending on the useMMap
    // parameter, possibly overridden.

    Value * const useMMap = b.getScalarField("useMMap");
    // if the fileDescriptor is 0, the file is stdin, use readSource kernel logic.
    Value * const fd = b.getScalarField("fileDescriptor");
    Value * const notStdIn = b.CreateICmpNE(fd, b.getInt32(STDIN_FILENO));
    Value * const tryMMap = b.CreateAnd(b.CreateIsNotNull(useMMap), notStdIn);
    b.CreateCondBr(tryMMap, checkFileSize, initializeRead);

    b.SetInsertPoint(checkFileSize);
    // If the fileSize is 0, we may have a virtual file such as /proc/cpuinfo
    Function * const fileSizeFn = b.getModule()->getFunction("file_size");
    assert (fileSizeFn);
    FunctionType * fTy = fileSizeFn->getFunctionType();
    Value * const fileSize = b.CreateCall(fTy, fileSizeFn, fd);
    Value * const emptyFile = b.CreateIsNull(fileSize);
    b.CreateUnlikelyCondBr(emptyFile, initializeRead, initializeMMap);

    b.SetInsertPoint(initializeMMap);
    MMapSourceKernel::generateInitializeMethod(b, mCodeUnitWidth, mStride);
    b.CreateBr(initializeDone);

    b.SetInsertPoint(initializeRead);
    // Ensure that readSource logic is used throughout.
    b.setScalarField("useMMap", ConstantInt::getNullValue(useMMap->getType()));
    ReadSourceKernel::generateInitializeMethod(b, mCodeUnitWidth, mStride);
    b.CreateBr(initializeDone);

    b.SetInsertPoint(initializeDone);
}

void FDSourceKernel::generateDoSegmentMethod(KernelBuilder & b) {
    BasicBlock * DoSegmentRead = b.CreateBasicBlock("DoSegmentRead");
    BasicBlock * DoSegmentMMap = b.CreateBasicBlock("DoSegmentMMap");
    BasicBlock * DoSegmentDone = b.CreateBasicBlock("DoSegmentDone");
    Value * const useMMap = b.CreateIsNotNull(b.getScalarField("useMMap"));
    b.CreateCondBr(useMMap, DoSegmentMMap, DoSegmentRead);
    b.SetInsertPoint(DoSegmentMMap);
    MMapSourceKernel::generateDoSegmentMethod(b, mCodeUnitWidth, mStride);
    b.CreateBr(DoSegmentDone);
    b.SetInsertPoint(DoSegmentRead);
    ReadSourceKernel::generateDoSegmentMethod(b, mCodeUnitWidth, mStride);
    b.CreateBr(DoSegmentDone);
    b.SetInsertPoint(DoSegmentDone);
}

Value * FDSourceKernel::generateExpectedOutputSizeMethod(KernelBuilder & b) {
    BasicBlock * finalizeRead = b.CreateBasicBlock("finalizeRead");
    BasicBlock * finalizeMMap = b.CreateBasicBlock("finalizeMMap");
    BasicBlock * finalizeDone = b.CreateBasicBlock("finalizeDone");
    Value * const useMMap = b.CreateIsNotNull(b.getScalarField("useMMap"));
    b.CreateCondBr(useMMap, finalizeMMap, finalizeRead);
    b.SetInsertPoint(finalizeMMap);
    Value * mmapVal = MMapSourceKernel::generateExpectedOutputSizeMethod(b, mCodeUnitWidth);
    b.CreateBr(finalizeDone);
    b.SetInsertPoint(finalizeRead);
    Value * readVal = ReadSourceKernel::generateExpectedOutputSizeMethod(b, mCodeUnitWidth);
    b.CreateBr(finalizeDone);
    b.SetInsertPoint(finalizeDone);
    PHINode * const resultPhi = b.CreatePHI(b.getSizeTy(), 2);
    resultPhi->addIncoming(mmapVal, finalizeMMap);
    resultPhi->addIncoming(readVal, finalizeRead);
    return resultPhi;
}

void FDSourceKernel::generateFinalizeMethod(KernelBuilder & b) {
    BasicBlock * finalizeRead = b.CreateBasicBlock("finalizeRead");
    BasicBlock * finalizeMMap = b.CreateBasicBlock("finalizeMMap");
    BasicBlock * finalizeDone = b.CreateBasicBlock("finalizeDone");
    Value * const useMMap = b.CreateIsNotNull(b.getScalarField("useMMap"));
    b.CreateCondBr(useMMap, finalizeMMap, finalizeRead);
    b.SetInsertPoint(finalizeMMap);
    MMapSourceKernel::freeBuffer(b, mCodeUnitWidth);
    b.CreateBr(finalizeDone);
    b.SetInsertPoint(finalizeRead);
    ReadSourceKernel::freeBuffer(b, mCodeUnitWidth);
    b.CreateBr(finalizeDone);
    b.SetInsertPoint(finalizeDone);
}

void FDSourceKernel::generateFinalizeThreadLocalMethod(KernelBuilder & b) {
    BasicBlock * finalizeRead = b.CreateBasicBlock("finalizeRead");
    BasicBlock * finalizeDone = b.CreateBasicBlock("finalizeDone");
    Value * const useMMap = b.CreateIsNotNull(b.getScalarField("useMMap"));
    b.CreateCondBr(useMMap, finalizeDone, finalizeRead);
    b.SetInsertPoint(finalizeRead);
    ReadSourceKernel::finalizeThreadLocalMethod(b, mCodeUnitWidth);
    b.CreateBr(finalizeDone);
    b.SetInsertPoint(finalizeDone);
}

void FDSourceKernel::linkExternalMethods(KernelBuilder & b) {
    MMapSourceKernel::generatLinkExternalFunctions(b);
    ReadSourceKernel::generatLinkExternalFunctions(b);
}

/// MEMORY SOURCE KERNEL

void MemorySourceKernel::generateInitializeMethod(KernelBuilder & b) {
    Value * const fileSource = b.getScalarField("fileSource");
    b.setBaseAddress("sourceBuffer", fileSource);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b.CreateAssert(fileSource, getName() + " fileSource cannot be null");
    }
    Value * const fileItems = b.getScalarField("fileItems");
    b.setCapacity("sourceBuffer", fileItems);
}

void MemorySourceKernel::generateDoSegmentMethod(KernelBuilder & b) {

    Value * const numOfStrides = b.getNumOfStrides();

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b.CreateAssert(b.CreateIsNotNull(numOfStrides),
                        "Internal error: %s.numOfStrides cannot be 0", b.GetString(getName()));
    }

    Value * const segmentItems = b.CreateMul(numOfStrides, b.getSize(getStride()));
    BasicBlock * const createTemporary = b.CreateBasicBlock("createTemporary");
    BasicBlock * const exit = b.CreateBasicBlock("exit");

    Value * const fileItems = b.getScalarField("fileItems");
    Value * const producedItems = b.getProducedItemCount("sourceBuffer");
    Value * const nextProducedItems = b.CreateAdd(producedItems, segmentItems);
    Value * const lastPage = b.CreateICmpULE(fileItems, nextProducedItems);
    b.CreateUnlikelyCondBr(lastPage, createTemporary, exit);

    b.SetInsertPoint(createTemporary);
    b.setTerminationSignal();
    b.setProducedItemCount("sourceBuffer", fileItems);
    b.CreateBr(exit);

    b.SetInsertPoint(exit);
}

Value * MemorySourceKernel::generateExpectedOutputSizeMethod(KernelBuilder & b) {
    return b.getScalarField("fileItems");
}

std::string makeSourceName(StringRef prefix, const unsigned fieldWidth, const unsigned numOfStreams = 1U) {
    std::string tmp;
    tmp.reserve(64);
    llvm::raw_string_ostream out(tmp);
    out << prefix << codegen::SegmentSize << '@' << fieldWidth;
    if (numOfStreams != 1) {
        out << ':' << numOfStreams;
    }
    out.flush();
    return tmp;
}

MMapSourceKernel::MMapSourceKernel(LLVMTypeSystemInterface & ts, Scalar * const fd, StreamSet * const outputStream)
: SegmentOrientedKernel(ts, makeSourceName("mmap_source", outputStream->getFieldWidth())
// input streams
,{}
// output streams
,{Binding{"sourceBuffer", outputStream, FixedRate(), { ManagedBuffer(), Linear() }}}
// input scalars
,{Binding{"fileDescriptor", fd}}
// output scalars
,{Binding{ts.getSizeTy(), "fileItems"}}
// internal scalars
,{})
, mCodeUnitWidth(outputStream->getFieldWidth()) {
    PointerType * const codeUnitPtrTy = ts.getIntNTy(mCodeUnitWidth)->getPointerTo();
    addInternalScalar(codeUnitPtrTy, "buffer");
    addAttribute(MustExplicitlyTerminate());
    addAttribute(SideEffecting());
    setStride(codegen::SegmentSize);
}

ReadSourceKernel::ReadSourceKernel(LLVMTypeSystemInterface & ts, Scalar * const fd, StreamSet * const outputStream)
: SegmentOrientedKernel(ts, makeSourceName("read_source", outputStream->getFieldWidth())
// input streams
,{}
// output streams
,{Binding{"sourceBuffer", outputStream, FixedRate(), { ManagedBuffer(), Linear() }}}
// input scalars
,{Binding{"fileDescriptor", fd}}
// output scalars
,{Binding{ts.getSizeTy(), "fileItems"}}
// internal scalars
,{})
, mCodeUnitWidth(outputStream->getFieldWidth()) {
    addAttribute(MustExplicitlyTerminate());
    addAttribute(SideEffecting());
    setStride(codegen::SegmentSize);
}



FDSourceKernel::FDSourceKernel(LLVMTypeSystemInterface & ts, Scalar * const useMMap, Scalar * const fd, StreamSet * const outputStream)
: SegmentOrientedKernel(ts, makeSourceName("FD_source", outputStream->getFieldWidth())
// input streams
,{}
// output stream
,{Binding{"sourceBuffer", outputStream, FixedRate(), { ManagedBuffer(), Linear() }}}
// input scalar
,{Binding{"useMMap", useMMap}
, Binding{"fileDescriptor", fd}}
// output scalar
,{Binding{ts.getSizeTy(), "fileItems"}}
// internal scalars
,{})
, mCodeUnitWidth(outputStream->getFieldWidth()) {
    PointerType * const codeUnitPtrTy = ts.getIntNTy(mCodeUnitWidth)->getPointerTo();
    addInternalScalar(codeUnitPtrTy, "buffer");
    addAttribute(MustExplicitlyTerminate());
    addAttribute(SideEffecting());
    setStride(codegen::SegmentSize);
}

MemorySourceKernel::MemorySourceKernel(LLVMTypeSystemInterface & ts, Scalar * fileSource, Scalar * fileItems, StreamSet * const outputStream)
: SegmentOrientedKernel(ts, makeSourceName("memory_source", outputStream->getFieldWidth(), outputStream->getNumElements()),
// input streams
{},
// output stream
{Binding{"sourceBuffer", outputStream, FixedRate(), { ManagedBuffer(), Linear() }}},
// input scalar
{Binding{"fileSource", fileSource}, Binding{"fileItems", fileItems}},
{},
// internal scalar
{}) {
    addAttribute(MustExplicitlyTerminate());
    addInternalScalar(fileSource->getType(), "ancillaryBuffer");
    setStride(codegen::SegmentSize);
}

}
