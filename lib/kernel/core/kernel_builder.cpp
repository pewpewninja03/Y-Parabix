#include <kernel/core/kernel_builder.h>
#include <kernel/core/kernel_compiler.h>
#include <toolchain/toolchain.h>
#include <kernel/core/streamset.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <boost/intrusive/detail/math.hpp>

using namespace llvm;

namespace kernel {

using PortType = Kernel::PortType;
using boost::intrusive::detail::floor_log2;

#define COMPILER (not_null<KernelCompiler *>(mCompiler))

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getHandle
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::getHandle() const noexcept {
    return COMPILER->getHandle();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getThreadLocalHandle
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::getThreadLocalHandle() const noexcept {
    return COMPILER->getThreadLocalHandle();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief hasScalarField
 ** ------------------------------------------------------------------------------------------------------------- */
bool KernelBuilder::hasScalarField(const StringRef fieldName) const {
    return COMPILER->hasScalarField(fieldName);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getScalarFieldPtr
 ** ------------------------------------------------------------------------------------------------------------- */
KernelBuilder::ScalarRef KernelBuilder::getScalarFieldPtr(const StringRef fieldName) {
    return COMPILER->getScalarFieldPtr(*this, fieldName);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getScalarField
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::getScalarField(const StringRef fieldName) {
    Type * ty; Value * ptr;
    std::tie(ptr, ty) = getScalarFieldPtr(fieldName);
    auto & DL = getModule()->getDataLayout();
    return CreateAlignedLoad(ty, ptr, DL.getABITypeAlign(ty).value(), fieldName);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setScalarField
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelBuilder::setScalarField(const StringRef fieldName, Value * const value) {
    auto sf = getScalarFieldPtr(fieldName);
    assert (value->getType() == sf.second);
    auto & DL = getModule()->getDataLayout();
    CreateAlignedStore(value, sf.first, DL.getABITypeAlign(sf.second).value());
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateMonitoredScalarFieldLoad
 ** ------------------------------------------------------------------------------------------------------------- */
LoadInst * KernelBuilder::CreateMonitoredScalarFieldLoad(const StringRef fieldName, Value * internalPtr) {
    Type * scalarTy;
    Value * scalarPtr;
    std::tie(scalarPtr, scalarTy) = getScalarFieldPtr(fieldName);
    Value * scalarEndPtr = CreateGEP(scalarTy, scalarPtr, getInt32(1));
    Value * internalEndPtr = CreateGEP(scalarTy, internalPtr, getInt32(1));
    Value * scalarAddr = CreatePtrToInt(scalarPtr, getSizeTy());
    Value * scalarEndAddr = CreatePtrToInt(scalarEndPtr, getSizeTy());
    Value * internalAddr = CreatePtrToInt(internalPtr, getSizeTy());
    Value * internalEndAddr = CreatePtrToInt(internalEndPtr, getSizeTy());
    Value * inBounds = CreateAnd(CreateICmpULE(scalarAddr, internalAddr), CreateICmpUGE(scalarEndAddr, internalEndAddr));
    __CreateAssert(inBounds, "Access (%" PRIx64 ",%" PRIx64 ") to scalar " + fieldName + " out of bounds (%" PRIx64 ",%" PRIx64 ").",
                   {scalarAddr, scalarEndAddr, internalAddr, internalEndAddr});
    return CreateLoad(scalarTy, internalPtr);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateMonitoredScalarFieldStore
 ** ------------------------------------------------------------------------------------------------------------- */
StoreInst * KernelBuilder::CreateMonitoredScalarFieldStore(const StringRef fieldName, Value * toStore, Value * internalPtr) {
    DataLayout DL(getModule());
    Type * scalarTy;
    Value * scalarPtr;
    std::tie(scalarPtr, scalarTy) = getScalarFieldPtr(fieldName);
    Value * scalarEndPtr = CreateGEP(scalarTy, scalarPtr, getInt32(1));
    Value * scalarAddr = CreatePtrToInt(scalarPtr, getSizeTy());
    Value * scalarEndAddr = CreatePtrToInt(scalarEndPtr, getSizeTy());
    Value * internalAddr = CreatePtrToInt(internalPtr, getSizeTy());
    Value * internalEndAddr = CreateAdd(internalAddr, getSize(DL.getTypeAllocSize(toStore->getType())));
    Value * inBounds = CreateAnd(CreateICmpULE(scalarAddr, internalAddr), CreateICmpUGE(scalarEndAddr, internalEndAddr));
    __CreateAssert(inBounds, "Store (%" PRIx64 ",%" PRIx64 ") to scalar " + fieldName + " out of bounds (%" PRIx64 ",%" PRIx64 ").",
                   {scalarAddr, scalarEndAddr, internalAddr, internalEndAddr});
    return CreateStore(toStore, internalPtr);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getTerminationSignal
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::getTerminationSignal() {
    Value * const ptr = COMPILER->getTerminationSignalPtr();
    if (ptr) {
        auto & dl = getModule()->getDataLayout();
        return CreateIsNotNull(CreateAlignedLoad(getSizeTy(), ptr, dl.getABITypeAlign(getSizeTy()).value()));
    } else {
        return getFalse();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setTerminationSignal
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelBuilder::setTerminationSignal(Value * const value) {
    auto & dl = getModule()->getDataLayout();
    Value * const ptr = COMPILER->getTerminationSignalPtr();
    if (LLVM_UNLIKELY(ptr == nullptr)) {
        report_fatal_error(StringRef(COMPILER->getName()) + " does not have CanTerminateEarly or MustExplicitlyTerminate set.");
    }
    CreateAlignedStore(value, ptr, dl.getABITypeAlign(getSizeTy()).value());
}

Value * KernelBuilder::getInputStreamBlockPtr(const StringRef name, Value * const streamIndex, Value * const blockOffset) {
    auto & dl = getModule()->getDataLayout();
    const auto & entry = COMPILER->getBinding(BindingType::StreamInput, name);
    Value * const processedPtr = COMPILER->getProcessedInputItemsPtr(entry.Index);
    Value * processed = CreateAlignedLoad(getSizeTy(), processedPtr, dl.getABITypeAlign(getSizeTy()).value());
    Value * blockIndex = CreateLShr(processed, floor_log2(getBitBlockWidth()));
    if (blockOffset) {
        blockIndex = CreateAdd(blockIndex, CreateZExtOrTrunc(blockOffset, blockIndex->getType()));
    }
    const StreamSetBuffer * const buf = COMPILER->getInputStreamSetBuffer(entry.Index);
    assert ("buffer is not accessible in this context!" && buf->getHandle());
    return buf->getStreamBlockPtr(*this, buf->getBaseAddress(*this), streamIndex, blockIndex);
}

Value * KernelBuilder::getInputStreamPackPtr(const StringRef name, Value * const streamIndex, Value * const packIndex, Value * const blockOffset) {
    auto & dl = getModule()->getDataLayout();
    const auto & entry = COMPILER->getBinding(BindingType::StreamInput, name);
    Value * const processedPtr = COMPILER->getProcessedInputItemsPtr(entry.Index);
    Value * processed = CreateAlignedLoad(getSizeTy(), processedPtr, dl.getABITypeAlign(getSizeTy()).value());
    Value * blockIndex = CreateLShr(processed, floor_log2(getBitBlockWidth()));
    if (blockOffset) {
        blockIndex = CreateAdd(blockIndex, CreateZExtOrTrunc(blockOffset, blockIndex->getType()));
    }
    const StreamSetBuffer * const buf = COMPILER->getInputStreamSetBuffer(entry.Index);
    assert ("buffer is not accessible in this context!" && buf->getHandle());
    return buf->getStreamPackPtr(*this, buf->getBaseAddress(*this), streamIndex, blockIndex, packIndex);
}

Value * KernelBuilder::loadInputStreamBlock(const StringRef name, Value * const streamIndex, Value * const blockOffset) {
    auto & dl = getModule()->getDataLayout();
    const auto bw = getBitBlockWidth();
    const auto & entry = COMPILER->getBinding(BindingType::StreamInput, name);
    Value * const processedPtr = COMPILER->getProcessedInputItemsPtr(entry.Index);
    Value * const processed = CreateAlignedLoad(getSizeTy(), processedPtr, dl.getABITypeAlign(getSizeTy()).value());
    Value * blockIndex = CreateLShr(processed, floor_log2(bw));
    if (blockOffset) {
        blockIndex = CreateAdd(blockIndex, CreateZExtOrTrunc(blockOffset, blockIndex->getType()));
    }
    const StreamSetBuffer * const buf = COMPILER->getInputStreamSetBuffer(entry.Index);
    assert ("buffer is not accessible in this context!" && buf->getHandle());
    Value * const ptr = buf->getStreamBlockPtr(*this, buf->getBaseAddress(*this), streamIndex, blockIndex);
    const auto dataWidth = (bw > 8) ? (bw >> 3) : 1U;
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts))) {
        Value * const count = buf->getStreamSetCount(*this);
        Value * const index = CreateZExtOrTrunc(streamIndex, count->getType());
        Value * const sanityCheck = CreateICmpULE(index, count);
        CreateAssert(sanityCheck, "stream index exceeds stream set count");
        Value * const start = CreateRoundDownRational(processed, getBitBlockWidth());
        Value * const end = COMPILER->getStreamSetAssertionInputItemCapacity(entry.Index);
        buf->assertAccessIsWithinStreamSetMemory(*this, GetString(name), ptr, dataWidth, start, end);
    }
    const auto unaligned = COMPILER->getInputStreamSetBinding(entry.Index).hasAttribute(Attribute::KindId::AllowsUnalignedAccess);
    return CreateAlignedLoad(getBitBlockType(), ptr, unaligned ? 1U : dataWidth);
}

Value * KernelBuilder::loadInputStreamPack(const StringRef name, Value * const streamIndex, Value * const packIndex, Value * const blockOffset) {
    auto & dl = getModule()->getDataLayout();
    const auto bw = getBitBlockWidth();
    const auto & entry = COMPILER->getBinding(BindingType::StreamInput, name);

    Value * const processedPtr = COMPILER->getProcessedInputItemsPtr(entry.Index);
    Value * const processed = CreateAlignedLoad(getSizeTy(), processedPtr, dl.getABITypeAlign(getSizeTy()).value());
    Value * blockIndex = CreateLShr(processed, floor_log2(bw));
    if (blockOffset) {
        blockIndex = CreateAdd(blockIndex, CreateZExtOrTrunc(blockOffset, blockIndex->getType()));
    }
    const StreamSetBuffer * const buf = COMPILER->getInputStreamSetBuffer(entry.Index);
    assert ("buffer is not accessible in this context!" && buf->getHandle());
    Value * const ptr = buf->getStreamPackPtr(*this, buf->getBaseAddress(*this), streamIndex, blockIndex, packIndex);
    const auto dataWidth = (bw > 8) ? (bw >> 3) : 1U;
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts))) {
        Value * const count = buf->getStreamSetCount(*this);
        Value * const index = CreateZExtOrTrunc(streamIndex, count->getType());
        Value * const sanityCheck = CreateICmpULE(index, count);
        CreateAssert(sanityCheck, "stream index exceeds stream set count");
        Value * const start = CreateRoundDownRational(processed, getBitBlockWidth());
        Value * const end = COMPILER->getStreamSetAssertionInputItemCapacity(entry.Index);
        buf->assertAccessIsWithinStreamSetMemory(*this, GetString(name), ptr, dataWidth, start, end);
    }
    const auto unaligned = COMPILER->getInputStreamSetBinding(entry.Index).hasAttribute(Attribute::KindId::AllowsUnalignedAccess);
    return CreateAlignedLoad(getBitBlockType(), ptr, unaligned ? 1U : dataWidth);
}

Value * KernelBuilder::getInputStreamSetCount(const StringRef name) {
    const StreamSetBuffer * const buf = COMPILER->getInputStreamSetBuffer(name);
    return buf->getStreamSetCount(*this);
}

Value * KernelBuilder::getOutputStreamBlockPtr(const StringRef name, Value * streamIndex, Value * const blockOffset) {
    const auto & entry = COMPILER->getBinding(BindingType::StreamOutput, name);
    Value * const producedPtr = COMPILER->getProducedOutputItemsPtr(entry.Index);
    Value * blockIndex = CreateLShr(CreateAlignedLoad(getSizeTy(), producedPtr, sizeof(size_t)), floor_log2(getBitBlockWidth()));
    if (blockOffset) {
        blockIndex = CreateAdd(blockIndex, CreateZExtOrTrunc(blockOffset, blockIndex->getType()));
    }
    const StreamSetBuffer * const buf = COMPILER->getOutputStreamSetBuffer(entry.Index);
    assert ("buffer is not accessible in this context!" && buf->getHandle());
    return buf->getStreamBlockPtr(*this, buf->getBaseAddress(*this), streamIndex, blockIndex);
}

Value * KernelBuilder::getOutputStreamPackPtr(const StringRef name, Value * streamIndex, Value * packIndex, Value * blockOffset) {
    const auto & entry = COMPILER->getBinding(BindingType::StreamOutput, name);
    Value * const producedPtr = COMPILER->getProducedOutputItemsPtr(entry.Index);
    Value * blockIndex = CreateLShr(CreateAlignedLoad(getSizeTy(), producedPtr, sizeof(size_t)), floor_log2(getBitBlockWidth()));
    if (blockOffset) {
        blockIndex = CreateAdd(blockIndex, CreateZExtOrTrunc(blockOffset, blockIndex->getType()));
    }
    const StreamSetBuffer * const buf = COMPILER->getOutputStreamSetBuffer(entry.Index);
    assert ("buffer is not accessible in this context!" && buf->getHandle());
    return buf->getStreamPackPtr(*this, buf->getBaseAddress(*this), streamIndex, blockIndex, packIndex);
}

StoreInst * KernelBuilder::storeOutputStreamBlock(const StringRef name, Value * streamIndex, Value * blockOffset, Value * toStore) {

    auto & dl = getModule()->getDataLayout();

    const auto bw = getBitBlockWidth();
    const auto & entry = COMPILER->getBinding(BindingType::StreamOutput, name);
    Value * const producedPtr = COMPILER->getProducedOutputItemsPtr(entry.Index);
    Value * produced = CreateAlignedLoad(getSizeTy(), producedPtr, dl.getABITypeAlign(getSizeTy()).value());
    Value * blockIndex = CreateLShr(produced, floor_log2(bw));
    if (blockOffset) {
        blockIndex = CreateAdd(blockIndex, CreateZExtOrTrunc(blockOffset, blockIndex->getType()));
    }
    const StreamSetBuffer * const buf = COMPILER->getOutputStreamSetBuffer(entry.Index);
    assert ("buffer is not accessible in this context!" && buf->getHandle());
    Value * const ptr = buf->getStreamBlockPtr(*this, buf->getBaseAddress(*this), streamIndex, blockIndex);
    const auto dataWidth = (bw > 8) ? (bw >> 3) : 1U;
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts))) {
        Value * const count = buf->getStreamSetCount(*this);
        Value * const index = CreateZExtOrTrunc(streamIndex, count->getType());
        Value * const sanityCheck = CreateICmpULE(index, count);
        CreateAssert(sanityCheck, "stream index exceeds stream set count");
        Value * const start = CreateRoundDownRational(produced, getBitBlockWidth());
        Value * const end = COMPILER->getStreamSetAssertionOutputItemCapacity(entry.Index);
        buf->assertAccessIsWithinStreamSetMemory(*this, GetString(name), ptr, dataWidth, start, end);
    }
    const auto unaligned = COMPILER->getOutputStreamSetBinding(entry.Index).hasAttribute(Attribute::KindId::AllowsUnalignedAccess);
    return CreateAlignedStore(toStore, ptr, unaligned ? 1U : dataWidth);
}

StoreInst * KernelBuilder::storeOutputStreamPack(const StringRef name, Value * streamIndex, Value * packIndex, Value * blockOffset, Value * toStore) {

    auto & dl = getModule()->getDataLayout();
    const auto bw = getBitBlockWidth();
    const auto & entry = COMPILER->getBinding(BindingType::StreamOutput, name);
    Value * const producedPtr = COMPILER->getProducedOutputItemsPtr(entry.Index);
    Value * produced = CreateAlignedLoad(getSizeTy(), producedPtr, dl.getABITypeAlign(getSizeTy()).value());
    Value * blockIndex = CreateLShr(produced, floor_log2(bw));
    if (blockOffset) {
        blockIndex = CreateAdd(blockIndex, CreateZExtOrTrunc(blockOffset, blockIndex->getType()));
    }
    const StreamSetBuffer * const buf = COMPILER->getOutputStreamSetBuffer(entry.Index);
    assert ("buffer is not accessible in this context!" && buf->getHandle());
    Value * const ptr = buf->getStreamPackPtr(*this, buf->getBaseAddress(*this), streamIndex, blockIndex, packIndex);
    const auto dataWidth = (bw > 8) ? (bw >> 3) : 1U;
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts))) {
        Value * const count = buf->getStreamSetCount(*this);
        Value * const index = CreateZExtOrTrunc(streamIndex, count->getType());
        Value * const sanityCheck = CreateICmpULE(index, count);
        CreateAssert(sanityCheck, "stream index exceeds stream set count");
        Value * const start = CreateRoundDownRational(produced, getBitBlockWidth());
        Value * const end = COMPILER->getStreamSetAssertionOutputItemCapacity(entry.Index);
        buf->assertAccessIsWithinStreamSetMemory(*this, GetString(name), ptr, dataWidth, start, end);
    }
    const auto unaligned = COMPILER->getOutputStreamSetBinding(entry.Index).hasAttribute(Attribute::KindId::AllowsUnalignedAccess);
    return CreateAlignedStore(toStore, ptr, unaligned ? 1U : dataWidth);
}

Value * KernelBuilder::getOutputStreamSetCount(const StringRef name) {
    const StreamSetBuffer * const buf = COMPILER->getOutputStreamSetBuffer(name);
    return buf->getStreamSetCount(*this);
}

Value * KernelBuilder::getRawInputPointer(const StringRef name, Value * absolutePosition) {
    const StreamSetBuffer * const buf = COMPILER->getInputStreamSetBuffer(name);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts))) {
        Value * const sanityCheck = CreateICmpEQ(buf->getStreamSetCount(*this), getSize(1));
        CreateAssert(sanityCheck, "stream index must be explicit");
    }
    return buf->getRawItemPointer(*this, getSize(0), absolutePosition);
}

Value * KernelBuilder::getRawInputPointer(const StringRef name, Value * const streamIndex, Value * absolutePosition) {
    const StreamSetBuffer * const buf = COMPILER->getInputStreamSetBuffer(name);
    return buf->getRawItemPointer(*this, streamIndex, absolutePosition);
}

Value * KernelBuilder::readRawInputPointer(Type * ty, const StringRef name, Value * absolutePosition) {
    const auto & binding = COMPILER->getBinding(BindingType::StreamInput, name);
    const StreamSetBuffer * const buf = COMPILER->getInputStreamSetBuffer(binding.Index);
    Value * ptr = buf->getRawItemPointer(*this, getSize(0), absolutePosition);
    auto & dl = getModule()->getDataLayout();
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts))) {
        Value * const sanityCheck = CreateICmpEQ(buf->getStreamSetCount(*this), getSize(1));
        CreateAssert(sanityCheck, "stream index must be explicit");
        Value * startPtr = COMPILER->getProcessedInputItemsPtr(binding.Index);
        Value * const start = CreateAlignedLoad(getSizeTy(), startPtr, dl.getABITypeAlign(getSizeTy()).value());
        Value * const end = COMPILER->getStreamSetAssertionInputItemCapacity(binding.Index);
        buf->assertAccessIsWithinStreamSetMemory(*this, GetString(name), ptr, getTypeSize(dl, ty), start, end);
    }
    const auto fw = buf->getFieldWidth();
    if (fw < 8) {
        IntegerType * const int8Ty = getInt8Ty();
        ptr = CreatePointerCast(ptr, int8Ty->getPointerTo(buf->getAddressSpace()));
        Value * val = CreateZExt(CreateAlignedLoad(int8Ty, ptr, 1), getSizeTy());
        const auto fieldsPerByte = 8U / fw;
        Value * pos = CreateZExtOrTrunc(absolutePosition, getSizeTy());
        pos = CreateAnd(pos, getSize(fieldsPerByte - 1));
        pos = CreateShl(pos, getSize(floor_log2(fw)));
        val = CreateLShr(val, pos);
        val = CreateAnd(val, getSize((1UL << fw) - 1UL));
        return CreateTrunc(val, ty);
    } else {
        ptr = CreatePointerCast(ptr, ty->getPointerTo(buf->getAddressSpace()));
        const auto dataWidth = fw > 8 ? fw >> 3 : 1U;
        const auto alignment = boost::gcd<size_t>(dl.getABITypeAlign(ty).value(), dataWidth);
        return CreateAlignedLoad(ty, ptr, alignment);
    }
}

Value * KernelBuilder::readRawInputPointer(Type * ty, const StringRef name, Value * const streamIndex, Value * absolutePosition) {
    const auto & binding = COMPILER->getBinding(BindingType::StreamInput, name);
    const StreamSetBuffer * const buf = COMPILER->getInputStreamSetBuffer(binding.Index);
    auto & dl = getModule()->getDataLayout();
    Value * ptr = buf->getRawItemPointer(*this, streamIndex, absolutePosition);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts))) {
        Value * const count = buf->getStreamSetCount(*this);
        Value * const index = CreateZExtOrTrunc(streamIndex, count->getType());
        Value * const sanityCheck = CreateICmpULE(index, count);
        CreateAssert(sanityCheck, "stream index exceeds stream set count");
        Value * const startPtr = COMPILER->getProcessedInputItemsPtr(binding.Index);
        Value * const start = CreateAlignedLoad(getSizeTy(), startPtr, dl.getABITypeAlign(getSizeTy()).value());
        Value * const end = COMPILER->getStreamSetAssertionInputItemCapacity(binding.Index);
        buf->assertAccessIsWithinStreamSetMemory(*this, GetString(name), ptr, getTypeSize(dl, ty), start, end);
    }

    const auto fw = buf->getFieldWidth();
    if (fw < 8) {
        IntegerType * const int8Ty = getInt8Ty();
        ptr = CreatePointerCast(ptr, int8Ty->getPointerTo(buf->getAddressSpace()));
        Value * val = CreateZExt(CreateAlignedLoad(int8Ty, ptr, 1), getSizeTy());
        const auto fieldsPerByte = 8U / fw;
        Value * pos = CreateZExtOrTrunc(absolutePosition, getSizeTy());
        pos = CreateAnd(pos, getSize(fieldsPerByte - 1));
        pos = CreateShl(pos, getSize(floor_log2(fw)));
        val = CreateLShr(val, pos);
        val = CreateAnd(val, getSize((1UL << fw) - 1UL));
        return CreateTrunc(val, ty);
    } else {
        ptr = CreatePointerCast(ptr, ty->getPointerTo(buf->getAddressSpace()));
        const auto dataWidth = fw > 8 ? fw >> 3 : 1U;
        const auto alignment = boost::gcd<size_t>(dl.getABITypeAlign(ty).value(), dataWidth);
        return CreateAlignedLoad(ty, ptr, alignment);
    }


}

Value * KernelBuilder::getRawOutputPointer(const StringRef name, Value * absolutePosition) {
    const StreamSetBuffer * const buf = COMPILER->getOutputStreamSetBuffer(name);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts))) {
        Value * const sanityCheck = CreateICmpEQ(buf->getStreamSetCount(*this), getSize(1));
        CreateAssert(sanityCheck, "stream index must be explicit");
    }
    return buf->getRawItemPointer(*this, getSize(0), absolutePosition);
}

Value * KernelBuilder::getRawOutputPointer(const StringRef name, Value * const streamIndex, Value * absolutePosition) {
    const StreamSetBuffer * const buf = COMPILER->getOutputStreamSetBuffer(name);
    return buf->getRawItemPointer(*this, streamIndex, absolutePosition);
}

Value * KernelBuilder::writeRawOutputPointer(const StringRef name, Value * absolutePosition, Value * value) {
    const auto & binding = COMPILER->getBinding(BindingType::StreamOutput, name);
    const StreamSetBuffer * const buf = COMPILER->getOutputStreamSetBuffer(binding.Index);
    Value * ptr = buf->getRawItemPointer(*this, getSize(0), absolutePosition);
    Type * const ty = value->getType();
    auto & dl = getModule()->getDataLayout();
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts))) {
        Value * const sanityCheck = CreateICmpEQ(buf->getStreamSetCount(*this), getSize(1));
        CreateAssert(sanityCheck, "stream index must be explicit");
        Value * const startPtr = COMPILER->getProducedOutputItemsPtr(binding.Index); assert (startPtr);
        Value * const start = CreateAlignedLoad(getSizeTy(), startPtr, dl.getABITypeAlign(getSizeTy()).value());
        Value * const end = COMPILER->getStreamSetAssertionOutputItemCapacity(binding.Index);
        buf->assertAccessIsWithinStreamSetMemory(*this, GetString(name), ptr, getTypeSize(dl, ty), start, end);
    }
    ptr = CreatePointerCast(ptr, ty->getPointerTo(buf->getAddressSpace()));
    const auto fw = buf->getFieldWidth();
    const auto dataWidth = fw > 8 ? fw >> 3 : 1U;
    const auto alignment = boost::gcd<size_t>(dl.getABITypeAlign(ty).value(), dataWidth);
    return CreateAlignedStore(value, ptr, alignment);
}

Value * KernelBuilder::writeRawOutputPointer(const StringRef name, Value * const streamIndex, Value * absolutePosition, Value * value) {
    const auto & binding = COMPILER->getBinding(BindingType::StreamOutput, name);
    const StreamSetBuffer * const buf = COMPILER->getOutputStreamSetBuffer(binding.Index);
    Value * ptr = buf->getRawItemPointer(*this, streamIndex, absolutePosition);
    Type * const ty = value->getType();
    auto & dl = getModule()->getDataLayout();
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts))) {
        Value * const count = buf->getStreamSetCount(*this);
        Value * const index = CreateZExtOrTrunc(streamIndex, count->getType());
        Value * const sanityCheck = CreateICmpULE(index, count);
        CreateAssert(sanityCheck, "stream index exceeds stream set count");
        Value * const startPtr = COMPILER->getProducedOutputItemsPtr(binding.Index); assert (startPtr);
        Value * const start = CreateAlignedLoad(getSizeTy(), startPtr, dl.getABITypeAlign(getSizeTy()).value());
        Value * const end = COMPILER->getStreamSetAssertionOutputItemCapacity(binding.Index);
        buf->assertAccessIsWithinStreamSetMemory(*this, GetString(name), ptr, getTypeSize(dl, ty), start, end);
    }
    ptr = CreatePointerCast(ptr, ty->getPointerTo(buf->getAddressSpace()));
    const auto fw = buf->getFieldWidth();
    const auto dataWidth = fw > 8 ? fw >> 3 : 1U;
    const auto alignment = boost::gcd<size_t>(dl.getABITypeAlign(ty).value(), dataWidth);
    return CreateAlignedStore(value, ptr, alignment);
}

Value * KernelBuilder::getBaseAddress(const StringRef name) {
    return COMPILER->getStreamSetBuffer(name)->getBaseAddress(*this);
}

void KernelBuilder::setBaseAddress(const StringRef name, Value * const addr) {
    return COMPILER->getStreamSetBuffer(name)->setBaseAddress(*this, addr);
}

Value * KernelBuilder::getCapacity(const StringRef name) {
    return COMPILER->getStreamSetBuffer(name)->getCapacity(*this);
}

void KernelBuilder::setCapacity(const StringRef name, Value * capacity) {
    COMPILER->getStreamSetBuffer(name)->setCapacity(*this, capacity);
}

void KernelBuilder::reserveCapacity(const StringRef name, Value * capacity) {
    const auto port = COMPILER->getStreamPort(name);
    if (LLVM_LIKELY(port.Type == PortType::Output)) {
        StreamSetBuffer * const buffer = COMPILER->getOutputStreamSetBuffer(port.Number);
        if (LLVM_LIKELY(isa<ManagedDynamicBuffer>(buffer))) {

            auto managedBuffer = cast<ManagedDynamicBuffer>(buffer);

            Module * const m = getModule();
            assert ("unspecified module" && m);
            auto & DL = m->getDataLayout();
            auto & C = getContext();
            IntegerType * const intPtrTy = DL.getIntPtrType(C);
            const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

            Value * const producedItemPtr = COMPILER->getProducedOutputItemsPtr(port.Number);
            Value * const producedItems = CreateAlignedLoad(intPtrTy, producedItemPtr,  intPtrTyAlign);
            Value * const consumedItems = COMPILER->getConsumedOutputItems(port.Number);

            SmallVector<char, 200> buf;
            raw_svector_ostream name(buf);

            const auto traceDynamicBuffers = codegen::DebugOptionIsSet(codegen::DebugFlags::TraceDynamicBuffers);

            name << "__KB_ManagedDynamicBuffer";
            if (buffer->isLinear()) {
                name << "Linear";
            }
            name << "_reserve_capacity" << buffer->getAddressSpace();
            if (LLVM_UNLIKELY(traceDynamicBuffers)) {
                name << 'T';
            }

            PointerType * const voidPtrTy = getVoidPtrTy();

            Function * f = m->getFunction(name.str());

            if (f == nullptr) {

                StructType * const handleTy = buffer->getHandleType(*this);
                PointerType * const handlePtrTy = handleTy->getPointerTo(buffer->getAddressSpace());

                SmallVector<Type *, 8> paramTypes(traceDynamicBuffers ? 7 : 4);
                paramTypes[0] = voidPtrTy; // shared struct ptr
                paramTypes[1] = intPtrTy;
                paramTypes[2] = intPtrTy;
                paramTypes[3] = intPtrTy;

                if (LLVM_UNLIKELY(traceDynamicBuffers)) {
                    paramTypes[4] = voidPtrTy;
                    paramTypes[5] = voidPtrTy;
                    paramTypes[6] = intPtrTy;
                }

                FunctionType * funcTy = FunctionType::get(getVoidTy(), paramTypes, false);

                const auto ip = saveIP();
                auto currentSharedHandle = managedBuffer->getHandle();
                f = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
                f->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);

                BasicBlock * const entry = BasicBlock::Create(C, "entry", f);
                BasicBlock * const expandInternalBuffer = BasicBlock::Create(C, "expandInternalBuffer", f);
                BasicBlock * const exit = BasicBlock::Create(C, "exit", f);

                SetInsertPoint(entry);

                auto arg = f->arg_begin();
                auto nextArg = [&]() {
                    assert (arg != f->arg_end());
                    Value * const v = &*arg;
                    std::advance(arg, 1);
                    return v;
                };

                Value * const handle = CreatePointerCast(nextArg(), handlePtrTy);
                managedBuffer->setHandle(handle);
                handle->setName("handle");
                Value * const produced = nextArg();
                produced->setName("produced");
                Value * const consumed = nextArg();
                consumed->setName("consumed");
                Value * const required = nextArg();
                required->setName("required");
                Value * reportExpansionCallback = nullptr;
                Value * pipelineHandle = nullptr;
                Value * portNum = nullptr;
                if (LLVM_UNLIKELY(traceDynamicBuffers)) {
                    reportExpansionCallback = nextArg();
                    reportExpansionCallback->setName("reportExpansionCallback");
                    pipelineHandle = nextArg();
                    pipelineHandle->setName("pipelineHandle");
                    portNum = nextArg();
                    portNum->setName("portNum");
                }
                assert (arg == f->arg_end());

                ConstantInt * const BLOCK_WIDTH = getSize(getBitBlockWidth());

                if (buffer->isLinear()) {
                    Value * const requiredChunks = CreateCeilUDiv(CreateAdd(produced, required), BLOCK_WIDTH);
                    Value * const capacityChunks = CreateUDiv(buffer->getCapacity(*this), BLOCK_WIDTH);
                    Value * const needsExpansion = CreateICmpUGT(requiredChunks, capacityChunks);
                    CreateUnlikelyCondBr(needsExpansion, expandInternalBuffer, exit);
                } else {
                    Value * const consumedChunks = CreateUDiv(consumed, BLOCK_WIDTH);
                    Value * const requiredChunks = CreateCeilUDiv(CreateAdd(produced, required), BLOCK_WIDTH);
                    Value * const capacityChunks = CreateUDiv(buffer->getInternalCapacity(*this), BLOCK_WIDTH);
                    CreateUnlikelyCondBr(CreateICmpUGT(requiredChunks, CreateAdd(capacityChunks, consumedChunks)), expandInternalBuffer, exit);
                }

                SetInsertPoint(expandInternalBuffer);
                managedBuffer->reserveCapacity(*this, produced, consumed, required, reportExpansionCallback, pipelineHandle, portNum);
                CreateBr(exit);

                SetInsertPoint(exit);
                CreateRetVoid();

                restoreIP(ip);
                managedBuffer->setHandle(currentSharedHandle);
            }

            SmallVector<Value *, 7> args(traceDynamicBuffers ? 7 : 4);
            args[0] = CreatePointerCast(managedBuffer->getHandle(), voidPtrTy);
            args[1] = producedItems;
            args[2] = consumedItems;
            args[3] = capacity;
            if (LLVM_UNLIKELY(traceDynamicBuffers)) {
                args[4] = COMPILER->getReportExpansionCallback();
                args[5] = COMPILER->getPipelineHandle();
                args[6] = getSize(port.Number);
            }
            
            CreateCall(f, args);
            return;
        }
    }
    report_fatal_error("Cannot call reserveCapacity on streamset " + name + ": is not a managed output streamset");
}

Value * KernelBuilder::getAvailableItemCount(const StringRef name) const noexcept {
    return COMPILER->getAvailableInputItems(name);
}

Value * KernelBuilder::getAccessibleItemCount(const StringRef name) const noexcept {
    return COMPILER->getAccessibleInputItems(name);
}

Value * KernelBuilder::getProcessedItemCount(const StringRef name) {
    auto & dl = getModule()->getDataLayout();
    return CreateAlignedLoad(getSizeTy(), COMPILER->getProcessedInputItemsPtr(name), dl.getABITypeAlign(getSizeTy()).value());
}

void KernelBuilder::setProcessedItemCount(const StringRef name, Value * value) {
    auto & dl = getModule()->getDataLayout();
    assert (value->getType() == getSizeTy());
    CreateAlignedStore(value, COMPILER->getProcessedInputItemsPtr(name), dl.getABITypeAlign(getSizeTy()).value());
}

Value * KernelBuilder::getProducedItemCount(const StringRef name) {
    auto & dl = getModule()->getDataLayout();
    return CreateAlignedLoad(getSizeTy(), COMPILER->getProducedOutputItemsPtr(name), dl.getABITypeAlign(getSizeTy()).value());
}

void KernelBuilder::setProducedItemCount(const StringRef name, Value * value) {
    auto & dl = getModule()->getDataLayout();
    assert (value->getType() == getSizeTy());
    CreateAlignedStore(value, COMPILER->getProducedOutputItemsPtr(name), dl.getABITypeAlign(getSizeTy()).value());
}

Value * KernelBuilder::getWritableOutputItems(const StringRef name) const noexcept {
    return COMPILER->getWritableOutputItems(name);
}

Value * KernelBuilder::getConsumedItemCount(const StringRef name) const noexcept {
    return COMPILER->getConsumedOutputItems(name);
}

// internal state

Value * KernelBuilder::getNumOfStrides() const noexcept {
    return COMPILER->getNumOfStrides();
}

Value * KernelBuilder::getExternalSegNo() const noexcept {
    return COMPILER->getExternalSegNo();
}

Value * KernelBuilder::isFinal() const noexcept {
    return COMPILER->isFinal();
}

// input streamset bindings

const Bindings & KernelBuilder::getInputStreamSetBindings() const noexcept {
    return COMPILER->getInputScalarBindings();
}

const Binding & KernelBuilder::getInputStreamSetBinding(const unsigned i) const noexcept {
    return COMPILER->getInputStreamSetBinding(i);
}

const Binding & KernelBuilder::getInputStreamSetBinding(const StringRef name) const noexcept {
    return COMPILER->getInputStreamSetBinding(name);
}

StreamSet * KernelBuilder::getInputStreamSet(const unsigned i) const noexcept {
    return COMPILER->getInputStreamSet(i);
}

StreamSet * KernelBuilder::getInputStreamSet(const StringRef name) const noexcept {
    return COMPILER->getInputStreamSet(name);
}

void KernelBuilder::setInputStreamSet(const StringRef name, StreamSet * value) noexcept {
    return COMPILER->setInputStreamSet(name, value);
}

unsigned KernelBuilder::getNumOfStreamInputs() const noexcept {
    return COMPILER->getNumOfStreamInputs();
}

// input streamsets

StreamSetBuffer * KernelBuilder::getInputStreamSetBuffer(const unsigned i) const noexcept {
    return COMPILER->getInputStreamSetBuffer(i);
}

StreamSetBuffer * KernelBuilder::getInputStreamSetBuffer(const StringRef name) const noexcept {
    return COMPILER->getInputStreamSetBuffer(name);
}

// output streamset bindings

const Bindings & KernelBuilder::getOutputStreamSetBindings() const noexcept {
    return COMPILER->getOutputStreamSetBindings();
}

const Binding & KernelBuilder::getOutputStreamSetBinding(const unsigned i) const noexcept {
    return COMPILER->getOutputStreamSetBinding(i);
}

const Binding & KernelBuilder::getOutputStreamSetBinding(const StringRef name) const noexcept {
    return COMPILER->getOutputStreamSetBinding(name);
}

StreamSet * KernelBuilder::getOutputStreamSet(const unsigned i) const noexcept {
    return COMPILER->getOutputStreamSet(i);
}

StreamSet * KernelBuilder::getOutputStreamSet(const StringRef name) const noexcept {
    return COMPILER->getOutputStreamSet(name);
}

void KernelBuilder::setOutputStreamSet(const StringRef name, StreamSet * value) noexcept {
    return COMPILER->setOutputStreamSet(name, value);
}

unsigned KernelBuilder::getNumOfStreamOutputs() const noexcept {
    return COMPILER->getNumOfStreamOutputs();
}

// output streamsets

StreamSetBuffer * KernelBuilder::getOutputStreamSetBuffer(const unsigned i) const noexcept {
    return COMPILER->getOutputStreamSetBuffer(i);
}

StreamSetBuffer * KernelBuilder::getOutputStreamSetBuffer(const StringRef name) const noexcept {
    return COMPILER->getOutputStreamSetBuffer(name);
}

// input scalar bindings

const Bindings & KernelBuilder::getInputScalarBindings() const noexcept {
    return COMPILER->getInputScalarBindings();
}

const Binding & KernelBuilder::getInputScalarBinding(const unsigned i) const noexcept {
    return COMPILER->getInputScalarBinding(i);
}

const Binding & KernelBuilder::getInputScalarBinding(const StringRef name) const noexcept {
    return COMPILER->getInputScalarBinding(name);
}

unsigned KernelBuilder::getNumOfScalarInputs() const noexcept {
    return COMPILER->getNumOfScalarInputs();
}

// input scalars

Scalar * KernelBuilder::getInputScalar(const unsigned i) noexcept {
    return COMPILER->getInputScalar(i);
}

Scalar * KernelBuilder::getInputScalar(const StringRef name) noexcept {
    return COMPILER->getInputScalar(name);
}

// output scalar bindings

const Bindings & KernelBuilder::getOutputScalarBindings() const noexcept {
    return COMPILER->getOutputScalarBindings();
}

const Binding & KernelBuilder::getOutputScalarBinding(const unsigned i) const noexcept {
    return COMPILER->getOutputScalarBinding(i);
}

const Binding & KernelBuilder::getOutputScalarBinding(const StringRef name) const noexcept {
    return COMPILER->getOutputScalarBinding(name);
}

unsigned KernelBuilder::getNumOfScalarOutputs() const noexcept {
    return COMPILER->getNumOfScalarOutputs();
}

// output scalars

Scalar * KernelBuilder::getOutputScalar(const unsigned i) noexcept {
    return COMPILER->getOutputScalar(i);
}

Scalar * KernelBuilder::getOutputScalar(const StringRef name) noexcept {
    return COMPILER->getOutputScalar(name);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateCeilAddRational
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::CreateCeilAddRational(Value * number, const Rational divisor, const Twine & Name) {
    if (LLVM_UNLIKELY(divisor.numerator() == 1 && divisor.denominator() == 1)) {
        return number;
    }
    Constant * const n = ConstantInt::get(number->getType(), divisor.numerator());
    if (LLVM_UNLIKELY(divisor.denominator() == 1)) {
        return CreateAdd(number, n, Name);
    }
    Constant * const d = ConstantInt::get(number->getType(), divisor.denominator());
    return CreateCeilUDiv(CreateAdd(CreateMul(number, d), n), d, Name);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateUDivRational
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::CreateUDivRational(Value * const number, const Rational divisor, const Twine & Name) {
    if (divisor.numerator() == 1 && divisor.denominator() == 1) {
        return number;
    }
    Constant * const n = ConstantInt::get(number->getType(), divisor.numerator());
    if (LLVM_LIKELY(divisor.denominator() == 1)) {
        return CreateUDiv(number, n, Name);
    } else {
        Constant * const d = ConstantInt::get(number->getType(), divisor.denominator());
        return CreateUDiv(CreateMul(number, d), n);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateCeilUDivRational
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::CreateCeilUDivRational(Value * number, const Rational divisor, const Twine & Name) {
    if (LLVM_UNLIKELY(divisor.numerator() == 1 && divisor.denominator() == 1)) {
        return number;
    }
    Constant * const n = ConstantInt::get(number->getType(), divisor.numerator());
    if (LLVM_UNLIKELY(divisor.denominator() != 1)) {
        Constant * const d = ConstantInt::get(number->getType(), divisor.denominator());
        number = CreateMul(number, d);
    }
    return CreateCeilUDiv(number, n, Name);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateMulRational
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::CreateMulRational(Value * const number, const Rational factor, const Twine & Name) {
    if (LLVM_UNLIKELY(factor.numerator() == 1 && factor.denominator() == 1)) {
        return number;
    }
    Constant * const n = ConstantInt::get(number->getType(), factor.numerator());
    if (LLVM_LIKELY(factor.denominator() == 1)) {
        return CreateMul(number, n, Name);
    } else {
        Constant * const d = ConstantInt::get(number->getType(), factor.denominator());
        return CreateUDiv(CreateMul(number, n), d, Name);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateCeilUMulRational
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::CreateCeilUMulRational(Value * const number, const Rational factor, const Twine & Name) {
    if (LLVM_LIKELY(factor.denominator() == 1)) {
        return CreateMulRational(number, factor, Name);
    }
    Constant * const n = ConstantInt::get(number->getType(), factor.numerator());
    Constant * const d = ConstantInt::get(number->getType(), factor.denominator());
    return CreateCeilUDiv(CreateMul(number, n), d, Name);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateURemRational
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::CreateURemRational(Value * const number, const Rational factor, const Twine & Name) {
    Constant * const n = ConstantInt::get(number->getType(), factor.numerator());
    if (LLVM_LIKELY(factor.denominator() == 1)) {
        return CreateURem(number, n, Name);
    }
    return CreateSub(number, CreateMulRational(CreateUDivRational(number, factor), factor), Name);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateRoundDownRational
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::CreateRoundDownRational(Value * const number, const Rational divisor, const Twine & Name) {
    Constant * const n = ConstantInt::get(number->getType(), divisor.numerator());
    if (divisor.denominator() == 1) {
        if (LLVM_UNLIKELY(divisor.numerator() == 1)) return number;
        return CBuilder::CreateRoundDown(number, n, Name);
    }
    Constant * const d = ConstantInt::get(number->getType(), divisor.denominator());
    return CreateUDiv(CBuilder::CreateRoundDown(CreateMul(number, d), n, Name), d);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateRoundUpRational
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelBuilder::CreateRoundUpRational(Value * const number, const Rational divisor, const Twine & Name) {
    Constant * const n = ConstantInt::get(number->getType(), divisor.numerator());
    if (divisor.denominator() == 1) {
        if (LLVM_UNLIKELY(divisor.numerator() == 1)) return number;
        return CBuilder::CreateRoundUp(number, n, Name);
    }
    Constant * const d = ConstantInt::get(number->getType(), divisor.denominator());
    return CreateUDiv(CBuilder::CreateRoundUp(CreateMul(number, d), n, Name), d);
}

#if 0

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeAddressableValue
 ** ------------------------------------------------------------------------------------------------------------- */
KernelBuilder::AddressableValue KernelBuilder::makeAddressableValue(Type * type, Value * value, Value * from, Value * to, const MemoryOrdering ordering) {
    if (LLVM_UNLIKELY(!!from ^ !!to)) {
        report_fatal_error("capture functions require either both or neither of the from/to positions to be set");
    }

    AddressableValue av;

    if (LLVM_UNLIKELY(value->getType()->isPointerTy())) {
        assert (type->getPointerTo() == value->getType());
        av.Address = value;
        av.From = from;
        av.To = to;
    } else {

        DataLayout DL(getModule());
        Type * vecTy = type;
        size_t rowCount = 1;
        assert (type->canLosslesslyBitCastTo(value->getType()));
        if (isa<ArrayType>(type)) {
            if (ordering == MemoryOrdering::ColumnMajor) {
                ArrayType * colTy = cast<ArrayType>(type);
                Type * rowTy = colTy->getArrayElementType();
                if (isa<ArrayType>(rowTy)) {
                    vecTy = rowTy->getArrayElementType();
                    rowCount = rowTy->getArrayNumElements();
                } else {
                    vecTy = rowTy;
                }
            } else if (ordering == MemoryOrdering::RowMajor) {
                ArrayType * rowTy = cast<ArrayType>(type);
                rowCount = rowTy->getArrayNumElements();
                Type * colTy = rowTy->getArrayElementType();
                if (isa<ArrayType>(colTy)) {
                    vecTy = colTy->getArrayElementType();
                } else {
                    vecTy = colTy;
                }
            }
            assert (rowCount > 0);
        }

        const auto a = getTypeSize(DL, vecTy) * 8;
        Type * elemTy = vecTy;
        if (LLVM_LIKELY(isa<VectorType>(vecTy))) {
            elemTy = cast<VectorType>(vecTy)->getElementType();
        }

        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
        const auto b = elemTy->getPrimitiveSizeInBits();
        #elif LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(16, 0, 0)
        const auto b = elemTy->getPrimitiveSizeInBits().getFixedSize();
        #else
        const auto b = elemTy->getPrimitiveSizeInBits().getFixedValue();
        #endif

        Rational R{a, b};
        assert (R.denominator() == 1);
        if (from && to) {
            av.From = from;
            av.To = to;
            if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
                CreateAssert(CreateICmpULT(av.To, getSize(R.numerator() * rowCount)), "from/to distance exceeds value size");
            }
        } else if (from || to) {
            report_fatal_error("capture functions require either both or neither of the from/to positions to be set");
        } else {
            av.From = getSize(0);
            av.To = getSize(R.numerator() * rowCount);
        }
        av.Address = CreateAllocaAtEntryPoint(type);
        CreateStore(value, av.Address);
    }

    return av;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief captureBitstream
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelBuilder::captureBitstream(StringRef streamName, Type * type, Value * bitstream, Value * from, Value * to, const MemoryOrdering ordering, const char zeroCh, const char oneCh) {
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {

        // TODO: should these be varadic functions that accept "indices" to allow users to decide how to name
        // capture functions that occur within loops? Problem with this is we cannot support this in general
        // and allow multi-threaded accesses since we cannot identify all of the streamNames at initialization.

        // Moreover, it would be difficult to display said iterated values logically without assuming a format.

        Constant * kernelName = GetString(mCompiler->getName());
        Constant * dataName = GetString(streamName);

        std::unique_ptr<KernelBuilder> tmp(this);
        mCompiler->registerIllustrator(tmp, kernelName, dataName,
                                       1, 1, 1, ordering,
                                       IllustratorTypeId::Bitstream, zeroCh, oneCh);

        const auto av = makeAddressableValue(type, bitstream, from, to, ordering);

        mCompiler->captureStreamData(tmp, GetString(mCompiler->getName()), GetString(streamName), getHandle(),
                                     getScalarField(KERNEL_ILLUSTRATOR_STRIDE_NUM),
                                     type, ordering, av.Address, av.From, av.To);
        tmp.release();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief captureBixNum
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelBuilder::captureBixNum(StringRef streamName, Type * type, Value * bixnum, Value * from, Value * to, const MemoryOrdering ordering, const char hexBase) {
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        Constant * kernelName = GetString(mCompiler->getName());
        Constant * dataName = GetString(streamName);

        std::unique_ptr<KernelBuilder> tmp(this);
        mCompiler->registerIllustrator(tmp, kernelName, dataName,
                                       1, 1, 1, ordering,
                                       IllustratorTypeId::BixNum, hexBase, '\0');

        const auto av = makeAddressableValue(type, bixnum, from, to, ordering);

        mCompiler->captureStreamData(tmp, kernelName, dataName, getHandle(),
                                     getScalarField(KERNEL_ILLUSTRATOR_STRIDE_NUM),
                                     type, ordering, av.Address, av.From, av.To);
        tmp.release();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief captureByteData
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelBuilder::captureByteData(StringRef streamName, Type * type, Value * byteData,  Value * from, Value * to, const MemoryOrdering ordering, const char nonASCIIsubstitute) {
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        // to capture this value, we want to automatically register it in the init phase
        std::unique_ptr<KernelBuilder> tmp(this);

        Constant * kernelName = GetString(mCompiler->getName());
        Constant * dataName = GetString(streamName);

        mCompiler->registerIllustrator(tmp, kernelName, dataName,
                                       1, 1, 1, ordering,
                                       IllustratorTypeId::ByteData, nonASCIIsubstitute, 0);

        const auto av = makeAddressableValue(type, byteData, from, to, ordering);
        mCompiler->captureStreamData(tmp, kernelName, dataName, getHandle(),
                                    getScalarField(KERNEL_ILLUSTRATOR_STRIDE_NUM),
                                    type, ordering, av.Address, av.From, av.To);
        tmp.release();
    }
}

#endif

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getKernelName
 ** ------------------------------------------------------------------------------------------------------------- */
std::string KernelBuilder::getKernelName() const noexcept {
    if (LLVM_UNLIKELY(mCompiler == nullptr)) {
        return "";
    }
    return mCompiler->getName();
}


#ifndef NDEBUG
/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isFromCurrentFunction
 ** ------------------------------------------------------------------------------------------------------------- */
bool isFromCurrentFunction(const KernelBuilder & b, const Value * const value, const bool allowNull) {
    if (value == nullptr) {
        assert ("value is null?" && allowNull);
        return allowNull;
    }
    if (LLVM_UNLIKELY(&b.getContext() != &value->getContext())) {
        assert (!"not from same context?");
        return false;
    }
    BasicBlock * const ip = b.GetInsertBlock(); assert (ip);
    if (isa<Constant>(value)) {
        return true;
    }
    const Function * const builderFunction = ip->getParent();
    assert (builderFunction);
    const Function * function = builderFunction;
    if (isa<Argument>(value)) {
        function = cast<Argument>(value)->getParent();
    } else if (isa<Instruction>(value)) {
        function = cast<Instruction>(value)->getParent()->getParent();
    }
    assert (function);
    assert ("not from same function?" && (builderFunction == function));
    return (builderFunction == function);
}
#endif

}
