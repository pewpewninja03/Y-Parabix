/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/core/streamset.h>

#include <kernel/core/kernel.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/core/kernel_builder.h>
#include <toolchain/toolchain.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/ADT/Twine.h>
#include <boost/intrusive/detail/math.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/integer/common_factor_rt.hpp>
#include <llvm/Analysis/ConstantFolding.h>

#include <array>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/mman.h>

#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

// #include <sys/memfd.h>

namespace llvm { class Constant; }
namespace llvm { class Function; }

using namespace llvm;
using namespace IDISA;
using IDISA::IDISA_Builder;

using boost::intrusive::detail::is_pow2;
using boost::intrusive::detail::floor_log2;

using Rational = kernel::StreamSetBuffer::Rational;

inline unsigned getPageSize() {
    return boost::interprocess::mapped_region::get_page_size();
}

inline static int create_memfd() {
#if defined(__FreeBSD__)
return shm_open(SHM_ANON, O_RDWR, 0);
#elif defined(__NR_memfd_create)
return syscall(__NR_memfd_create, "shm_anon", (unsigned int)(MFD_CLOEXEC));
#else
char nm[] = "/tmp/streamset-XXXXXX";
#if defined(__OpenBSD__)
const int fd = shm_mkstemp(nm);
if (shm_unlink(nm) != 0) {
    close(fd);
    return -1;
}
#else
const int fd = mkstemp(nm);
if (unlink(nm) != 0) {
    close(fd);
    return -1;
}
#endif
return fd;
#endif
}

uint8_t * make_circular_buffer(const size_t size, const size_t hasUnderflow) {

    assert (size > 0);
    assert ((size % getPageSize()) == 0);

    const auto memfd = create_memfd();
    if (memfd == -1) {
        SmallVector<char, 256> tmp;
        raw_svector_ostream msg(tmp);
        msg << "failed to create memfd (" << (size_t)errno << ')';
        report_fatal_error(msg.str());
    }

    if (ftruncate(memfd, size) == -1) {
        report_fatal_error(Twine{"failed to size mmap buffer to ", std::to_string(size)});
    }

    assert (hasUnderflow == 0 || hasUnderflow == 1);

    const size_t m = 2 + hasUnderflow;

    uint8_t * base = (uint8_t*)mmap(NULL, m * size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (base == MAP_FAILED) {
        report_fatal_error("failed to allocate virtual address space");
    }

    for (size_t i = 0; i < m; ++i) {
        void * p = mmap(base + (i * size), size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, memfd, 0);
        if (p == MAP_FAILED) {
            report_fatal_error("failed to map virtual address space");
        }
    }
    close(memfd);
    return base + (hasUnderflow * size);
}

void destroy_circular_buffer(uint8_t * base, const size_t size, const size_t hasUnderflow) {
    if (LLVM_UNLIKELY(base == nullptr)) {
        return;
    }
    const size_t n = hasUnderflow ? 1 : 0;
    const size_t m = n + 2;
    const auto ptr = base - (n * size);
    munmap(ptr, m * size);

}

#define ANON_MMAP_SIZE (2ULL * 1048576ULL)

inline bool isConstantOne(const Value * const index) {
    return isa<ConstantInt>(index) ? cast<ConstantInt>(index)->isOne() : false;
}

inline bool isCapacityGuaranteed(const Value * const index, const size_t capacity) {
    return isa<ConstantInt>(index) ? cast<ConstantInt>(index)->getLimitedValue() < capacity : false;
}

namespace kernel {

using Rational = KernelBuilder::Rational;

[[noreturn]] void unsupported(const char * const function, const char * const bufferType) {
    report_fatal_error(StringRef{function} + " is not supported by " + bufferType + "Buffers");
}

LLVM_READNONE inline unsigned getItemWidth(const Type * ty ) {
    if (LLVM_LIKELY(isa<ArrayType>(ty))) {
        ty = ty->getArrayElementType();
    }
    ty = cast<FixedVectorType>(ty)->getElementType();
    return cast<IntegerType>(ty)->getBitWidth();
}

LLVM_READNONE inline size_t getArraySize(const Type * ty) {
    if (LLVM_LIKELY(isa<ArrayType>(ty))) {
        return ty->getArrayNumElements();
    } else {
        return 1;
    }
}

void StreamSetBuffer::assertValidStreamIndex(KernelBuilder & b, Value * streamIndex) const {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value * const count = getStreamSetCount(b);
        Value * const index = b.CreateZExtOrTrunc(streamIndex, count->getType());
        Value * const withinSet = b.CreateICmpULT(index, count);
        b.CreateAssert(withinSet, "out-of-bounds stream access: %i of %i", index, count);
    }
}

Value * StreamSetBuffer::getStreamBlockPtr(KernelBuilder & b, Value * const baseAddress, Value * const streamIndex, Value * const blockIndex) const {
   // assertValidStreamIndex(b, streamIndex);
    return b.CreateInBoundsGEP(mType, baseAddress, {blockIndex, streamIndex});
}

Value * StreamSetBuffer::getStreamPackPtr(KernelBuilder & b, Value * const baseAddress, Value * const streamIndex, Value * blockIndex, Value * const packIndex) const {
   // assertValidStreamIndex(b, streamIndex);
    return b.CreateInBoundsGEP(mType, baseAddress, {blockIndex, streamIndex, packIndex});
}

Value * StreamSetBuffer::getStreamSetCount(KernelBuilder & b) const {
    size_t count = 1;
    if (isa<ArrayType>(getBaseType())) {
        count = getBaseType()->getArrayNumElements();
    }
    return b.getSize(count);
}

bool StreamSetBuffer::isEmptySet() const {
    return getArraySize(mBaseType) == 0;
}

bool StreamSetBuffer::isSingleElementStreamSet() const {
    return getArraySize(mBaseType) <= 1;
}

unsigned StreamSetBuffer::getFieldWidth() const {
    return getItemWidth(mBaseType);
}

/**
 * @brief getRawItemPointer
 *
 * get a raw pointer the iN field at position absoluteItemPosition of the stream number streamIndex of the stream set.
 * In the case of a stream whose fields are less than one byte (8 bits) in size, the pointer is to the containing byte.
 * The type of the pointer is i8* for fields of 8 bits or less, otherwise iN* for N-bit fields.
 */
Value * StreamSetBuffer::getRawItemPointer(KernelBuilder & b, Value * streamIndex, Value * absolutePosition) const {
    Type * const elemTy = cast<ArrayType>(mBaseType)->getElementType();
    Type * itemTy = cast<VectorType>(elemTy)->getElementType();
    #if LLVM_VERSION_CODE < LLVM_VERSION_CODE(12, 0, 0)
    const unsigned itemWidth = itemTy->getPrimitiveSizeInBits();
    #else
    const unsigned itemWidth = itemTy->getPrimitiveSizeInBits().getFixedSize();
    #endif
    IntegerType * const sizeTy = b.getSizeTy();
    absolutePosition = b.CreateZExt(absolutePosition, sizeTy);
    streamIndex = b.CreateZExt(streamIndex, sizeTy);

    Value * pos = nullptr;
    Value * addr = nullptr;
    Value * const streamCount = getStreamSetCount(b);
    if (LLVM_LIKELY(isConstantOne(streamCount))) {
        addr = getBaseAddress(b);
        pos = absolutePosition;
        if (!isLinear()) {
            pos = b.CreateURem(pos, getCapacity(b));
        }
    } else {
        Constant * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
        Value * blockIndex = b.CreateUDiv(absolutePosition, BLOCK_WIDTH);
        addr = getStreamBlockPtr(b, getBaseAddress(b), streamIndex, blockIndex);
        pos = b.CreateURem(absolutePosition, BLOCK_WIDTH);
    }
    if (LLVM_UNLIKELY(itemWidth < 8)) {
        const Rational itemsPerByte{8, itemWidth};
        pos = b.CreateUDivRational(pos, itemsPerByte);
        itemTy = b.getInt8Ty();
    }
    addr = b.CreatePointerCast(addr, itemTy->getPointerTo(mAddressSpace));
    return b.CreateInBoundsGEP(itemTy, addr, pos);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief resolveType
 ** ------------------------------------------------------------------------------------------------------------- */
Type * StreamSetBuffer::resolveType(KernelBuilder & b, Type * const streamSetType) {
    unsigned numElements = 1;
    Type * type = streamSetType;
    if (LLVM_LIKELY(type->isArrayTy())) {
        numElements = type->getArrayNumElements();
        type = type->getArrayElementType();
    }
    if (LLVM_LIKELY(type->isVectorTy() && cast<FixedVectorType>(type)->getNumElements() == 0)) {
        type = cast<FixedVectorType>(type)->getElementType();
        if (LLVM_LIKELY(type->isIntegerTy())) {
            const auto fieldWidth = cast<IntegerType>(type)->getBitWidth();
            type = b.getBitBlockType();
            if (fieldWidth != 1) {
                type = ArrayType::get(type, fieldWidth);
            }
            return ArrayType::get(type, numElements);
        }
    }
    std::string tmp;
    raw_string_ostream out(tmp);
    streamSetType->print(out);
    out << " is an unvalid stream set buffer type.";
    report_fatal_error(Twine(out.str()));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isItemAlignedAccessWithinStreamSetMemory
 ** ------------------------------------------------------------------------------------------------------------- */
void StreamSetBuffer::assertAccessIsWithinStreamSetMemory(KernelBuilder & b, Constant * name, Value * ptr, const size_t size, llvm::Value * const start, llvm::Value * const end) const {
#if 0
    // TODO: to do this right, we need a "total space" parameter passed from the pipeline to this

    assert (codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts));

    Value * startPtr = getRawItemPointer(b, b.getSize(0), start);
    Value * const streamCount = b.CreateSub(getStreamSetCount(b), b.getSize(1));
    Value * endPtr = getRawItemPointer(b, streamCount, b.CreateRoundUpRational(end, b.getBitBlockWidth()));
    auto & dl = b.getModule()->getDataLayout();
    IntegerType * const intPtrTy = dl.getIntPtrType(b.getContext());
    startPtr = b.CreatePtrToInt(startPtr, intPtrTy);
    ptr = b.CreatePtrToInt(ptr, intPtrTy);
    Value * valid = b.CreateICmpULE(startPtr, ptr);
    Value * outPtr = b.CreateAdd(ptr, ConstantInt::get(intPtrTy, size)); assert (size > 0);
    endPtr = b.CreatePtrToInt(endPtr, intPtrTy);
    valid = b.CreateAnd(valid, b.CreateICmpULE(outPtr, endPtr));
    b.CreateAssert(valid, "streamset \"%s\" memory access [%" PRIx64 ",%" PRIx64 ") is outside of valid memory boundaries [%" PRIx64 ",%" PRIx64 ")",
                   name, ptr, outPtr, startPtr, endPtr);
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkFunctions
 ** ------------------------------------------------------------------------------------------------------------- */
constexpr const char __MAKE_CIRCULAR_BUFFER[] =    "__ss_mcb";
constexpr auto __MAKE_CIRCULAR_BUFFER_LENGTH = std::string_view(__MAKE_CIRCULAR_BUFFER).size();
constexpr const char __DESTROY_CIRCULAR_BUFFER[] = "__ss_dcb";
constexpr auto __DESTROY_CIRCULAR_BUFFER_LENGTH = std::string_view(__DESTROY_CIRCULAR_BUFFER).size();

void StreamSetBuffer::linkFunctions(KernelBuilder & b) {
    b.LinkFunction(StringRef{__MAKE_CIRCULAR_BUFFER, __MAKE_CIRCULAR_BUFFER_LENGTH}, make_circular_buffer);
    b.LinkFunction(StringRef{__DESTROY_CIRCULAR_BUFFER, __DESTROY_CIRCULAR_BUFFER_LENGTH}, destroy_circular_buffer);
}

// External Buffer

StructType * ExternalBuffer::getHandleType(KernelBuilder & b) const {
    if (mHandleType == nullptr) {
        FixedArray<Type *, 2> fields;
        fields[0] = getPointerType();
        fields[1] = b.getSizeTy();
        mHandleType = StructType::get(b.getContext(), fields);
    }
    return mHandleType;
}

void ExternalBuffer::allocateBuffer(KernelBuilder & /* b */, Value * const /* capacityMultiplier */) {
    unsupported("allocateBuffer", "External");
}

void ExternalBuffer::releaseBuffer(KernelBuilder & /* b */) const {
    // this buffer is not responsible for free-ing th data associated with it
}

void ExternalBuffer::destroyBuffer(KernelBuilder & b, Value * baseAddress, Value *capacity) const {

}

void ExternalBuffer::setBaseAddress(KernelBuilder & b, Value * const addr) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    Value * const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(BaseAddress)});
    b.CreateStore(b.CreatePointerBitCastOrAddrSpaceCast(addr, getPointerType()), p);
}

Value * ExternalBuffer::getBaseAddress(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling getBaseAddress");
    Value * const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(BaseAddress)});
    return b.CreateAlignedLoad(getPointerType(), p, sizeof(void*));
}

void ExternalBuffer::setCapacity(KernelBuilder & b, Value * const capacity) const {
    assert (mHandle && "has not been set prior to calling setCapacity");
    Value *  const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(EffectiveCapacity)});
    b.CreateAlignedStore(b.CreateZExt(capacity, b.getSizeTy()), p, sizeof(size_t));
}

Value * ExternalBuffer::getCapacity(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling getCapacity");
    Value * const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(EffectiveCapacity)});
    return b.CreateLoad(b.getSizeTy(), p);
}

Value * ExternalBuffer::getInternalCapacity(KernelBuilder & b) const {
    return getCapacity(b);
}

Value * ExternalBuffer::modByCapacity(KernelBuilder & /* b */, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    return offset;
}

Value * ExternalBuffer::getLinearlyAccessibleItems(KernelBuilder & b, Value * const fromPosition, Value * const totalItems) const {
    return b.CreateSub(totalItems, fromPosition);
}

Value * ExternalBuffer::getLinearlyWritableItems(KernelBuilder & b, Value * const fromPosition, Value * const /* consumed */) const {
    assert (fromPosition);
    Value * const capacity = getCapacity(b);
    assert (fromPosition->getType() == capacity->getType());
    return b.CreateSub(capacity, fromPosition);
}

Value * ExternalBuffer::getVirtualBasePtr(KernelBuilder & b, Value * baseAddress, Value * const /* transferredItems */) const {
    Constant * const sz_ZERO = b.getSize(0);
    Value * const addr = StreamSetBuffer::getStreamBlockPtr(b, baseAddress, sz_ZERO, sz_ZERO);
    return b.CreatePointerCast(addr, getPointerType());
}

inline void ExternalBuffer::assertValidBlockIndex(KernelBuilder & b, Value * blockIndex) const {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value * const blockCount = b.CreateCeilUDiv(getCapacity(b), b.getSize(b.getBitBlockWidth()));
        blockIndex = b.CreateZExtOrTrunc(blockIndex, blockCount->getType());
        Value * const withinCapacity = b.CreateICmpULT(blockIndex, blockCount);
        b.CreateAssert(withinCapacity, "blockIndex exceeds buffer capacity");
    }
}

Value * ExternalBuffer::requiresExpansion(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    unsupported("requiresExpansion", "External");
}

void ExternalBuffer::linearCopyBack(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    unsupported("linearCopyBack", "External");
}

Value * ExternalBuffer::expandBuffer(KernelBuilder & /* b */, Value * /* produced */, Value * /* consumed */, Value * const /* required */) const  {
    unsupported("expandBuffer", "External");
}

Value * ExternalBuffer::getMallocAddress(KernelBuilder & /* b */) const {
    unsupported("getMallocAddress", "External");
}

// Internal Buffer

Value * InternalBuffer::getStreamBlockPtr(KernelBuilder & b, Value * const baseAddress, Value * const streamIndex, Value * const blockIndex) const {
    Value * offset = nullptr;
    if (mLinear) {
        offset = blockIndex;
    } else {
        offset = modByCapacity(b, blockIndex);
    }
    return StreamSetBuffer::getStreamBlockPtr(b, baseAddress, streamIndex, offset);
}

Value * InternalBuffer::getStreamPackPtr(KernelBuilder & b, Value * const baseAddress, Value * const streamIndex, Value * const blockIndex, Value * const packIndex) const {
    Value * offset = nullptr;
    if (mLinear) {
        offset = blockIndex;
    } else {
        offset = modByCapacity(b, blockIndex);
    }
    return StreamSetBuffer::getStreamPackPtr(b, baseAddress, streamIndex, offset, packIndex);
}

Value * InternalBuffer::getVirtualBasePtr(KernelBuilder & b, Value * const baseAddress, Value * const transferredItems) const {
    Constant * const sz_ZERO = b.getSize(0);
    Value * baseBlockIndex = nullptr;
    if (mLinear) {
        // NOTE: the base address of a linear buffer is always the virtual base ptr; just return it.
        baseBlockIndex = sz_ZERO;
    } else {
        Constant * const LOG_2_BLOCK_WIDTH = b.getSize(floor_log2(b.getBitBlockWidth()));
        Value * const blockIndex = b.CreateLShr(transferredItems, LOG_2_BLOCK_WIDTH);
        baseBlockIndex = b.CreateSub(modByCapacity(b, blockIndex), blockIndex);
    }
    Value * addr = StreamSetBuffer::getStreamBlockPtr(b, baseAddress, sz_ZERO, baseBlockIndex);
    return b.CreatePointerCast(addr, getPointerType());
}

Value * InternalBuffer::getLinearlyAccessibleItems(KernelBuilder & b, Value * const processedItems, Value * const totalItems) const {
    return b.CreateSub(totalItems, processedItems);
}

Value * InternalBuffer::getLinearlyWritableItems(KernelBuilder & b, Value * const producedItems, Value * const consumedItems) const {
    Value * const capacity = getCapacity(b);
    if (LLVM_UNLIKELY(mLinear)) {
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            Value * const valid = b.CreateICmpULE(producedItems, capacity);
            b.CreateAssert(valid, "produced item count (%" PRIu64 ") exceeds capacity (%" PRIu64 ").",
                            producedItems, capacity);
        }
        return b.CreateSub(capacity, producedItems);
     } else {
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            Value * const valid = b.CreateICmpULE(consumedItems, producedItems);
            b.CreateAssert(valid, "consumed item count (%" PRIu64 ") exceeds produced (%" PRIu64 ").",
                            consumedItems, producedItems);
        }
        Value * const unconsumedItems = b.CreateSub(producedItems, consumedItems);
        return b.CreateSub(capacity, unconsumedItems);
    }
}

// Dynamic Buffer

StructType * DynamicBuffer::getHandleType(KernelBuilder & b) const {
    if (mHandleType == nullptr) {
        auto & C = b.getContext();
        PointerType * const typePtr = getPointerType();
        IntegerType * const sizeTy = b.getSizeTy();

        if (mLinear) {
            FixedArray<Type *, LinearFields> types;
            types[LinearMallocedAddress] = typePtr;
            types[LinearInternalCapacity] = sizeTy;
            types[LinearBaseAddress] = typePtr;
            types[LinearEffectiveCapacity] = sizeTy;
            mHandleType = StructType::get(C, types);
        } else {
            FixedArray<Type *, CircularFields> types;
            types[CircularAddressSelector] = sizeTy;
            types[CircularBaseAddress] = typePtr;
            types[CircularInternalCapacity] = sizeTy;
            types[CircularSecondaryBaseAddress] = typePtr;
            types[CircularSecondaryInternalCapacity] = sizeTy;
            mHandleType = StructType::get(C, types);
        }

        assert (mHandleType);

    }
    return mHandleType;
}

void DynamicBuffer::allocateBuffer(kernel::KernelBuilder & b, Value * const capacity) {
    assert (mHandle && "has not been set prior to calling allocateBuffer");
    // note: when adding extensible stream sets, make sure to set the initial count here.
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);

    Value * const handle = getHandle();

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b.CreateAssert(capacity, "Dynamic buffer capacity cannot be 0.");
    }

    Module * m = b.getModule();
    DataLayout DL(m);

    const auto typeSize = b.getTypeSize(DL, mType);
    const auto pageSize = getPageSize();
    Rational stridesPerPage{pageSize, typeSize};
    Value * pageAlignedCapacity = b.CreateRoundUpRational(capacity, stridesPerPage.numerator());
    Value * capacityBytes = b.CreateMul(b.getSize(typeSize), pageAlignedCapacity);

    if (mLinear) {

        Value * baseAddress = b.CreateAlignedMalloc(capacityBytes, pageSize);

        indices[1] = b.getInt32(LinearBaseAddress);
        assert (mHandleType->getElementType(LinearBaseAddress)->isPointerTy());
        Value * const baseAddressField = b.CreateInBoundsGEP(mHandleType, handle, indices);

        indices[1] = b.getInt32(LinearInternalCapacity);
        assert (mHandleType->getElementType(LinearInternalCapacity)->isIntegerTy());
        Value * const capacityField = b.CreateInBoundsGEP(mHandleType, handle, indices);

        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * const initialField = b.CreateInBoundsGEP(mHandleType, handle, indices);

        indices[1] = b.getInt32(LinearEffectiveCapacity);
        Value * const effCapacityField = b.CreateInBoundsGEP(mHandleType, handle, indices);

        b.CreateStore(baseAddress, baseAddressField);
        b.CreateStore(pageAlignedCapacity, capacityField);
        b.CreateStore(baseAddress, initialField);
        b.CreateStore(pageAlignedCapacity, effCapacityField);

    } else {


        FixedArray<Value *, 2> makeArgs;
        makeArgs[0] = capacityBytes;
        makeArgs[1] = b.getSize(mHasUnderflow);
        Function * makeBuffer = m->getFunction(__MAKE_CIRCULAR_BUFFER); assert (makeBuffer);
        Value * baseAddress = b.CreateCall(makeBuffer, makeArgs);

        indices[1] = b.getInt32(CircularBaseAddress);
        assert (mHandleType->getElementType(CircularBaseAddress)->isPointerTy());
        Value * const baseAddressField = b.CreateInBoundsGEP(mHandleType, handle, indices);

        indices[1] = b.getInt32(CircularInternalCapacity);
        assert (mHandleType->getElementType(CircularInternalCapacity)->isIntegerTy());
        Value * const capacityField = b.CreateInBoundsGEP(mHandleType, handle, indices);

        indices[1] = b.getInt32(CircularSecondaryBaseAddress);
        Value * const secAddressField = b.CreateInBoundsGEP(mHandleType, handle, indices);

        indices[1] = b.getInt32(CircularSecondaryInternalCapacity);
        Value * const secCapacityField = b.CreateInBoundsGEP(mHandleType, handle, indices);

        b.CreateStore(baseAddress, baseAddressField);
        b.CreateStore(pageAlignedCapacity, capacityField);
        b.CreateStore(baseAddress, secAddressField);
        b.CreateStore(pageAlignedCapacity, secCapacityField);

    }

}

void DynamicBuffer::releaseBuffer(KernelBuilder & b) const {
    /* Free the dynamically allocated buffer(s). */
    Value * const handle = getHandle();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    if (mLinear) {
        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * const baseAddressField = b.CreateInBoundsGEP(mHandleType, handle, indices);
        Value * const baseAddress = b.CreateLoad(getPointerType(), baseAddressField);
        b.CreateFree(baseAddress);
        b.CreateStore(ConstantPointerNull::get(getPointerType()), baseAddressField);
    } else {
        indices[1] = b.getInt32(CircularAddressSelector);
        Value * const selectorField = b.CreateInBoundsGEP(mHandleType, handle, indices);
        Value * const selector = b.CreateIsNotNull(b.CreateLoad(b.getSizeTy(), selectorField));

        indices[1] = b.getInt32(CircularSecondaryBaseAddress);
        Value * const secBaseAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(CircularBaseAddress);
        Value * const primBaseAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const baseAddressField = b.CreateSelect(selector, secBaseAddr, primBaseAddr);
        Value * const baseAddress = b.CreateLoad(getPointerType(), baseAddressField);


        indices[1] = b.getInt32(CircularSecondaryInternalCapacity);
        Value * const secCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(CircularInternalCapacity);
        Value * const primCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const capacityField = b.CreateSelect(selector, secCapacityField, primCapacityField);

        Value * const capacity = b.CreateLoad(b.getSizeTy(), capacityField);
        destroyBuffer(b, baseAddress, capacity);
        b.CreateStore(ConstantPointerNull::get(getPointerType()), baseAddressField);
    }

}

void DynamicBuffer::destroyBuffer(KernelBuilder & b, Value * baseAddress, Value * capacity) const {
    if (mLinear) {
        b.CreateFree(baseAddress);
    } else {
        FixedArray<Value *, 3> destroyArgs;
        destroyArgs[0] = b.CreatePointerCast(baseAddress, b.getInt8PtrTy());
        destroyArgs[1] = b.CreateMul(b.getTypeSize(mType), capacity);
        destroyArgs[2] = b.getSize(mHasUnderflow);
        b.CreateCall(b.getModule()->getFunction(__DESTROY_CIRCULAR_BUFFER), destroyArgs);
    }
}

void DynamicBuffer::setBaseAddress(KernelBuilder & /* b */, Value * /* addr */) const {
    assert (false);
    unsupported("setBaseAddress", "Dynamic");
}

Value * DynamicBuffer::getBaseAddress(KernelBuilder & b) const {
    assert (getHandle());
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * ptr = nullptr;
    if (mLinear) {
        indices[1] = b.getInt32(LinearBaseAddress);
        ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    } else {
        indices[1] = b.getInt32(CircularAddressSelector);
        Value * const selectorField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const selector = b.CreateIsNotNull(b.CreateLoad(b.getSizeTy(), selectorField));
        indices[1] = b.getInt32(CircularSecondaryBaseAddress);
        Value * const secBaseAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(CircularBaseAddress);
        Value * const primBaseAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        ptr = b.CreateSelect(selector, secBaseAddr, primBaseAddr);
    }
    return b.CreateLoad(getPointerType(), ptr);
}

Value * DynamicBuffer::getMallocAddress(KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * ptr = nullptr;
    if (mLinear) {
        indices[1] = b.getInt32(LinearMallocedAddress);
        ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    } else {
        indices[1] = b.getInt32(CircularAddressSelector);
        Value * const selectorField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const selector = b.CreateIsNotNull(b.CreateLoad(b.getSizeTy(), selectorField));
        indices[1] = b.getInt32(CircularSecondaryBaseAddress);
        Value * const secBaseAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(CircularBaseAddress);
        Value * const primBaseAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        ptr = b.CreateSelect(selector, secBaseAddr, primBaseAddr);
    }
    return b.CreateLoad(getPointerType(), ptr);
}

Value * DynamicBuffer::modByCapacity(KernelBuilder & b, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    if (mLinear) {
        return offset;
    } else {
        assert (getHandle());
        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(0);
        indices[1] = b.getInt32(CircularAddressSelector);
        Value * const selectorField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const selector = b.CreateIsNotNull(b.CreateLoad(b.getSizeTy(), selectorField));


        indices[1] = b.getInt32(CircularSecondaryInternalCapacity);
        Value * const secCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(CircularInternalCapacity);
        Value * const primCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const capacityPtr = b.CreateSelect(selector, secCapacityField, primCapacityField);

        Value * const capacity = b.CreateLoad(b.getSizeTy(), capacityPtr);
        return b.CreateURem(offset, capacity);
    }
}

Value * DynamicBuffer::getCapacity(KernelBuilder & b) const {
    assert (mHandle && mHandleType);

    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * capacity = nullptr;
    if (mLinear) {
        assert (mHandleType->getNumElements() == LinearFields);
        indices[1] = b.getInt32(LinearEffectiveCapacity);
        Value * ptr = b.CreateInBoundsGEP(mHandleType, mHandle, indices);
        capacity = b.CreateLoad(b.getSizeTy(), ptr);
    } else {

        assert (mHandleType->getNumElements() == CircularFields);
        indices[1] = b.getInt32(CircularAddressSelector);
        Value * const selectorField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const selector = b.CreateIsNotNull(b.CreateLoad(b.getSizeTy(), selectorField));

        indices[1] = b.getInt32(CircularSecondaryInternalCapacity);
        Value * const secCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(CircularInternalCapacity);
        Value * const primCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const capacityPtr = b.CreateSelect(selector, secCapacityField, primCapacityField);
        capacity = b.CreateLoad(b.getSizeTy(), capacityPtr);
    }
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    return b.CreateMul(capacity, BLOCK_WIDTH);
}

Value * DynamicBuffer::getInternalCapacity(KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * capacity = nullptr;
    if (mLinear) {
        indices[1] = b.getInt32(LinearInternalCapacity);
        Value * ptr = b.CreateInBoundsGEP(mHandleType, mHandle, indices);
        capacity = b.CreateLoad(b.getSizeTy(), ptr);
    } else {
        assert (mHandleType->getNumElements() == CircularFields);
        indices[1] = b.getInt32(CircularAddressSelector);
        Value * const selectorField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const selector = b.CreateIsNotNull(b.CreateLoad(b.getSizeTy(), selectorField));

        indices[1] = b.getInt32(CircularSecondaryInternalCapacity);
        Value * const secCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(CircularInternalCapacity);
        Value * const primCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const capacityPtr = b.CreateSelect(selector, secCapacityField, primCapacityField);
        capacity = b.CreateLoad(b.getSizeTy(), capacityPtr);
    }
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    return b.CreateMul(capacity, BLOCK_WIDTH);
}

void DynamicBuffer::setCapacity(KernelBuilder & /* b */, Value * /* capacity */) const {
    unsupported("setCapacity", "Dynamic");
}

Value * DynamicBuffer::requiresExpansion(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {

    if (mLinear) {

        const auto blockWidth = b.getBitBlockWidth();
        assert (is_pow2(blockWidth));

        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(0);

        ConstantInt * const BLOCK_WIDTH = b.getSize(blockWidth);

        Value * const consumedChunks = b.CreateUDiv(consumed, BLOCK_WIDTH);

        indices[1] = b.getInt32(LinearBaseAddress);
        Value * const virtualBaseField = b.CreateInBoundsGEP(mHandleType, mHandle, indices);
        Value * const virtualBase = b.CreateLoad(getPointerType(), virtualBaseField);
        Value * startOfUsedBuffer = b.CreateInBoundsGEP(mType, virtualBase, consumedChunks);
        DataLayout DL(b.getModule());
        Type * const intPtrTy = DL.getIntPtrType(virtualBase->getType());
        startOfUsedBuffer = b.CreatePtrToInt(startOfUsedBuffer, intPtrTy);

        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * const mallocedAddressField = b.CreateInBoundsGEP(mHandleType, mHandle, indices);
        Value * const mallocedAddress = b.CreateLoad(getPointerType(), mallocedAddressField);
        Value * const newPos = b.CreateAdd(produced, required);
        Value * const newChunks = b.CreateCeilUDiv(newPos, BLOCK_WIDTH);
        Value * const requiredChunks = b.CreateSub(newChunks, consumedChunks);
        Value * requiresUpToPosition = b.CreateInBoundsGEP(mType, mallocedAddress, requiredChunks);
        requiresUpToPosition = b.CreatePtrToInt(requiresUpToPosition, intPtrTy);

        return b.CreateICmpUGT(requiresUpToPosition, startOfUsedBuffer);

    } else { // Circular
        unsupported("requiresExpansion", "CircularDynamic");
    }

}

void DynamicBuffer::linearCopyBack(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {

    if (mLinear) {

        SmallVector<char, 200> buf;
        raw_svector_ostream name(buf);

        assert ("unspecified module" && b.getModule());

        name << "__DynamicBuffer_linearCopyBack";

        Type * ty = getBaseType();
        const auto streamCount = ty->getArrayNumElements();
        name << streamCount << 'x';
        ty = ty->getArrayElementType();
        ty = cast<FixedVectorType>(ty)->getElementType();
        const auto itemWidth = ty->getIntegerBitWidth();
        name << itemWidth << '@' << mAddressSpace;


        Value * const myHandle = getHandle();

        Module * const m = b.getModule();

        Function * func = m->getFunction(name.str());
        if (func == nullptr) {

            IntegerType * const sizeTy = b.getSizeTy();
            FixedArray<Type *, 3> params;
            params[0] = myHandle->getType();
            params[1] = sizeTy;
            params[2] = sizeTy;

            FunctionType * funcTy = FunctionType::get(b.getVoidTy(), params, false);

            const auto ip = b.saveIP();

            LLVMContext & C = m->getContext();
            func = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
            if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
                #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
                func->setHasUWTable();
                #else
                func->setUWTableKind(UWTableKind::Default);
                #endif
            }
            b.SetInsertPoint(BasicBlock::Create(C, "entry", func));

            auto arg = func->arg_begin();
            auto nextArg = [&]() {
                assert (arg != func->arg_end());
                Value * const v = &*arg;
                std::advance(arg, 1);
                return v;
            };

            Value * const handle = nextArg();
            handle->setName("handle");
            Value * const produced = nextArg();
            produced->setName("produced");
            Value * const consumed = nextArg();
            consumed->setName("consumed");
            assert (arg == func->arg_end());

            setHandle(handle);

            const auto blockWidth = b.getBitBlockWidth();
            assert (is_pow2(blockWidth));
            const auto blockSize = blockWidth / 8;
            const auto sizeTyWidth = sizeTy->getBitWidth() / 8;

            ConstantInt * const BLOCK_WIDTH = b.getSize(blockWidth);
            Constant * const CHUNK_SIZE = b.getTypeSize(mType);

            FixedArray<Value *, 2> indices;
            indices[0] = b.getInt32(0);

            Value * const consumedChunks = b.CreateUDiv(consumed, BLOCK_WIDTH);
            Value * const producedChunks = b.CreateCeilUDiv(produced, BLOCK_WIDTH);

            Value * const unconsumedChunks = b.CreateSub(producedChunks, consumedChunks);
            Value * const bytesToCopy = b.CreateMul(unconsumedChunks, CHUNK_SIZE);

            indices[1] = b.getInt32(LinearBaseAddress);
            Value * const virtualBaseField = b.CreateInBoundsGEP(mHandleType, handle, indices);
            Value * const virtualBase = b.CreateLoad(getPointerType(), virtualBaseField);

            indices[1] = b.getInt32(LinearMallocedAddress);
            Value * const mallocedAddressField = b.CreateInBoundsGEP(mHandleType, handle, indices);
            Value * const mallocedAddress = b.CreateAlignedLoad(getPointerType(), mallocedAddressField, sizeTyWidth);

            Value * const unreadDataPtr = b.CreateInBoundsGEP(mType, virtualBase, consumedChunks);
            b.CreateMemCpy(mallocedAddress, unreadDataPtr, bytesToCopy, blockSize);
            Value * const newVirtualAddress = b.CreateGEP(mType, mallocedAddress, b.CreateNeg(consumedChunks));
            b.CreateAlignedStore(newVirtualAddress, virtualBaseField, sizeTyWidth);
            indices[1] = b.getInt32(LinearInternalCapacity);
            Value * const intCapacityField = b.CreateInBoundsGEP(mHandleType, handle, indices);
            Value * const internalCapacity = b.CreateAlignedLoad(b.getSizeTy(), intCapacityField, sizeTyWidth);


            Value * const effectiveCapacity = b.CreateSub(b.CreateAdd(consumedChunks, internalCapacity), unconsumedChunks);
            indices[1] = b.getInt32(LinearEffectiveCapacity);
            Value * const effCapacityField = b.CreateInBoundsGEP(mHandleType, handle, indices);
            b.CreateAlignedStore(effectiveCapacity, effCapacityField, sizeTyWidth);

            b.CreateRetVoid();

            b.restoreIP(ip);
            setHandle(myHandle);
        }

        FixedArray<Value *, 3> args;
        args[0] = myHandle;
        args[1] = produced;
        args[2] = consumed;
        b.CreateCall(func->getFunctionType(), func, args);

    }
}

inline size_t largestPowerOf2ThatDivides(const size_t n) {
    return (n & (~(n - 1)));
}

Value * DynamicBuffer::expandBuffer(KernelBuilder & b, Value * const produced, Value * const consumed, Value * const required) const {

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    assert ("unspecified module" && b.getModule());

    name << "__DynamicBuffer_";
    if (mLinear) {
        name << "linear";
    } else {
        name << "circular";
    }
    name << "Expand_";

    Type * ty = getBaseType();
    const auto streamCount = ty->getArrayNumElements();
    name << streamCount << 'x';
    ty = ty->getArrayElementType();
    ty = cast<FixedVectorType>(ty)->getElementType();
    const auto itemWidth = ty->getIntegerBitWidth();
    name << itemWidth << '@' << mAddressSpace;

    Value * const myHandle = getHandle();

    Module * const m = b.getModule();

    Function * func = m->getFunction(name.str());
    if (func == nullptr) {

        IntegerType * const sizeTy = b.getSizeTy();
        PointerType * const voidPtrTy = b.getVoidPtrTy();

        FixedArray<Type *, 2> retTypes;
        retTypes[0] = voidPtrTy;
        retTypes[1] = sizeTy;
        StructType * retValTy = StructType::create(b.getContext(), retTypes);

        FixedArray<Type *, 5> paramTypes;
        paramTypes[0] = myHandle->getType();
        paramTypes[1] = sizeTy;
        paramTypes[2] = sizeTy;
        paramTypes[3] = sizeTy;
        paramTypes[4] = sizeTy;

        FunctionType * funcTy = FunctionType::get(retValTy, paramTypes, false);

        const auto ip = b.saveIP();

        LLVMContext & C = m->getContext();
        func = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            func->setHasUWTable();
            #else
            func->setUWTableKind(UWTableKind::Default);
            #endif
        }
        b.SetInsertPoint(BasicBlock::Create(C, "entry", func));

        auto arg = func->arg_begin();
        auto nextArg = [&]() {
            assert (arg != func->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };

        Value * const handle = nextArg();
        handle->setName("handle");
        Value * const produced = nextArg();
        produced->setName("produced");
        Value * const consumed = nextArg();
        consumed->setName("consumed");
        Value * const required = nextArg();
        required->setName("required");
        Value * const hasUnderflow = nextArg();
        hasUnderflow->setName("hasUnderflow");
        assert (arg == func->arg_end());

        setHandle(handle);

        const auto blockWidth = b.getBitBlockWidth();
        assert (is_pow2(blockWidth));
        const auto blockSize = blockWidth / 8;

        ConstantInt * const BLOCK_WIDTH = b.getSize(blockWidth);
        Constant * const typeSize = b.getTypeSize(mType);

        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(0);


        Value * const consumedChunks = b.CreateUDiv(consumed, BLOCK_WIDTH);
        Value * const producedChunks = b.CreateCeilUDiv(produced, BLOCK_WIDTH);
        Value * const requiredCapacity = b.CreateAdd(produced, required);
        Value * const requiredChunks = b.CreateCeilUDiv(requiredCapacity, BLOCK_WIDTH);

        Value * const unconsumedChunks = b.CreateSub(producedChunks, consumedChunks);
        Value * const bytesToCopy = b.CreateMul(unconsumedChunks, typeSize);

        Value * const chunksToReserve = b.CreateSub(requiredChunks, consumedChunks);


        const auto sizeTyWidth = sizeTy->getBitWidth() / 8;


        StructType * handleTy = getHandleType(b);


        FixedArray<Value *, 2> retVals;

        if (mLinear) {

            indices[1] = b.getInt32(LinearBaseAddress);

            Value * const virtualBaseField = b.CreateInBoundsGEP(handleTy, handle, indices);
            Value * const virtualBase = b.CreateLoad(getPointerType(), virtualBaseField);

            indices[1] = b.getInt32(LinearInternalCapacity);
            Value * const intCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
            Value * const internalCapacity = b.CreateAlignedLoad(sizeTy, intCapacityField, sizeTyWidth);

            retVals[1] = internalCapacity;

            // newInternalCapacity tends to be 2x internalCapacity
            Value * const reserveCapacity = b.CreateAdd(chunksToReserve, internalCapacity);
            Value * const newInternalCapacity = b.CreateRoundUp(reserveCapacity, internalCapacity);


            Value * const mallocSize = b.CreateMul(newInternalCapacity, typeSize);
            Value * expandedBuffer = b.CreatePointerCast(b.CreatePageAlignedMalloc(mallocSize), getPointerType());

            Value * const unreadDataPtr = b.CreateInBoundsGEP(mType, virtualBase, consumedChunks);
            b.CreateMemCpy(expandedBuffer, unreadDataPtr, bytesToCopy, blockSize);

            b.CreateAlignedStore(newInternalCapacity, intCapacityField, sizeTyWidth);

            indices[1] = b.getInt32(LinearMallocedAddress);
            Value * const mallocedAddressField = b.CreateInBoundsGEP(handleTy, handle, indices);
            Value * const mallocedAddress = b.CreateAlignedLoad(getPointerType(), mallocedAddressField, sizeTyWidth);

            retVals[0] = b.CreatePointerCast(mallocedAddress, voidPtrTy);

            b.CreateAlignedStore(expandedBuffer, mallocedAddressField, sizeTyWidth);

            Value * const effectiveCapacity = b.CreateAdd(consumedChunks, newInternalCapacity);
            indices[1] = b.getInt32(LinearEffectiveCapacity);
            Value * const effCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);

            Value * const newVirtualAddress = b.CreateGEP(mType, expandedBuffer, b.CreateNeg(consumedChunks));

            b.CreateAlignedStore(newVirtualAddress, virtualBaseField, sizeTyWidth);
            b.CreateAlignedStore(effectiveCapacity, effCapacityField, sizeTyWidth);

        } else { // Circular

            Module * const m = b.getModule();

            indices[1] = b.getInt32(CircularAddressSelector);
            Value * const selectorField = b.CreateInBoundsGEP(mHandleType, handle, indices);
            Value * const selectorVal = b.CreateLoad(b.getSizeTy(), selectorField);
            Value * const selector = b.CreateIsNotNull(selectorVal);

            indices[1] = b.getInt32(CircularSecondaryBaseAddress);
            Value * const secBaseAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
            indices[1] = b.getInt32(CircularBaseAddress);
            Value * const primBaseAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
            Value * const baseAddrField = b.CreateSelect(selector, secBaseAddr, primBaseAddr);

            PointerType * const typePtrTy = mType->getPointerTo();

            Value * const virtualBase = b.CreateLoad(typePtrTy, baseAddrField);
            retVals[0] = b.CreatePointerCast(virtualBase, voidPtrTy);

            indices[1] = b.getInt32(CircularSecondaryInternalCapacity);
            Value * const secCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
            indices[1] = b.getInt32(CircularInternalCapacity);
            Value * const primCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);

            Value * const intCapacityField = b.CreateSelect(selector, secCapacityField, primCapacityField);

            Value * const internalCapacity = b.CreateAlignedLoad(sizeTy, intCapacityField, sizeTyWidth);
            retVals[1] = internalCapacity;

            // newInternalCapacity tends to be 2x internalCapacity
            Value * const reserveCapacity = b.CreateAdd(chunksToReserve, internalCapacity);
            Value * const newInternalCapacity = b.CreateRoundUp(reserveCapacity, internalCapacity);
            FixedArray<Value *, 2> makeArgs;
            makeArgs[0] = b.CreateMul(newInternalCapacity, typeSize);
            makeArgs[1] = b.getSize(mHasUnderflow ? 1U : 0U);

            Value * newBuffer = b.CreateCall(m->getFunction(__MAKE_CIRCULAR_BUFFER), makeArgs);
            newBuffer = b.CreatePointerCast(newBuffer, typePtrTy);

            Value * newBufferOffset = b.CreateURem(consumedChunks, newInternalCapacity);
            Value * const toWriteDataPtr = b.CreateInBoundsGEP(mType, newBuffer, newBufferOffset);


            Value * const oldBufferOffset = b.CreateURem(consumedChunks, internalCapacity);
            Value * const unreadDataPtr = b.CreateInBoundsGEP(mType, virtualBase, oldBufferOffset);

            auto & DL = m->getDataLayout();
            const auto typeSize = b.getTypeSize(DL, mType);
            assert (typeSize >= (b.getBitBlockWidth() / 8));
            const auto align = largestPowerOf2ThatDivides(typeSize);
            assert ((typeSize % align) == 0);
            assert (align >= (b.getBitBlockWidth() / 8));
            assert ((align % (b.getBitBlockWidth() / 8)) == 0);


            b.CreateMemCpy(toWriteDataPtr, unreadDataPtr, bytesToCopy, align);

            Value * const secVirtualBaseField = b.CreateSelect(selector, primBaseAddr, secBaseAddr);

            b.CreateAlignedStore(newBuffer, secVirtualBaseField, sizeTyWidth);

            Value * const secIntCapacityField = b.CreateSelect(selector, primCapacityField, secCapacityField);

            b.CreateAlignedStore(newInternalCapacity, secIntCapacityField, sizeTyWidth);

            // Without a 128-bit atomic store, this must be last thing set or we cannot guarantee the consumers
            // will always read a proper addr/capacity pair
            b.CreateAlignedStore(b.CreateXor(selectorVal, b.getSize(1)), selectorField, sizeTyWidth);

        }
        b.CreateAggregateRet(retVals.data(), 2);

        b.restoreIP(ip);
        setHandle(myHandle);
    }

    FixedArray<Value *, 5> args;
    args[0] = myHandle;
    args[1] = produced;
    args[2] = consumed;
    args[3] = required;
    args[4] = b.getSize(mHasUnderflow);
    return b.CreateCall(func->getFunctionType(), func, args);
}

// Managed Dynamic Buffer

StructType * ManagedDynamicBuffer::getHandleType(KernelBuilder & b) const {
    if (mHandleType == nullptr) {
        auto & C = b.getContext();
        PointerType * const typePtr = getPointerType();
        IntegerType * const sizeTy = b.getSizeTy();

        FixedArray<Type *, LinearFields> types;
        types[LinearMallocedAddress] = typePtr;
        types[LinearInternalCapacity] = sizeTy;
        types[LinearBaseAddress] = typePtr;
        types[LinearEffectiveCapacity] = sizeTy;
        mHandleType = StructType::get(C, types);

        assert (mHandleType);

    }
    return mHandleType;
}

Value * ManagedDynamicBuffer::getVirtualBasePtr(KernelBuilder & b, Value * const baseAddress, Value * const transferredItems) const {
    return b.CreatePointerCast(baseAddress, getPointerType());
}

void ManagedDynamicBuffer::allocateBuffer(KernelBuilder & b, Value * const capacityMultiplier) {
    assert (mHandle && "has not been set prior to calling allocateBuffer");

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    assert ("unspecified module" && b.getModule());

    name << "__ManagedDynamicBuffer_initial_alloc";

    Type * ty = getBaseType();
    const auto streamCount = ty->getArrayNumElements();
    name << streamCount << 'x';
    ty = ty->getArrayElementType();
    ty = cast<FixedVectorType>(ty)->getElementType();
    const auto itemWidth = ty->getIntegerBitWidth();
    name << itemWidth << '@' << mAddressSpace;

    assert (!mLinear);

    Module * const m = b.getModule();

    Function * f = m->getFunction(name.str());

    if (f == nullptr) {

        IntegerType * const sizeTy = b.getSizeTy();

        StructType * handleTy = getHandleType(b);
        PointerType * const handlePtrTy = handleTy->getPointerTo(mAddressSpace);

        auto & DL = m->getDataLayout();

        PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);

        Constant * const nullAddrPtr = ConstantPointerNull::get(addrPtrTy);

        const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();

        FixedArray<Type *, 2> paramTypes;
        paramTypes[0] = handlePtrTy;
        paramTypes[1] = sizeTy;

        FunctionType * funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

        const auto ip = b.saveIP();

        f = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            f->setHasUWTable();
            #else
            f->setUWTableKind(UWTableKind::Default);
            #endif
        } else {
            f->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);
        }

        LLVMContext & C = m->getContext();
        BasicBlock * const entry = BasicBlock::Create(C, "entry", f);
        BasicBlock * const allocBuffers = BasicBlock::Create(C, "allocBuffers", f);
        BasicBlock * const exit = BasicBlock::Create(C, "exit", f);

        b.SetInsertPoint(entry);

        auto arg = f->arg_begin();
        auto nextArg = [&]() {
            assert (arg != f->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };

        Value * const handle = nextArg();
        handle->setName("handle");
        Value * capacity = nextArg();
        capacity->setName("capacity");
        assert (arg == f->arg_end());


        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(0);
        indices[1] = b.getInt32(LinearBaseAddress);
        assert (mHandleType->getElementType(LinearBaseAddress)->isPointerTy());
        Value * const baseAddressField = b.CreateInBoundsGEP(handleTy, handle, indices);
        // If the user has filled in a base address in the init function, assume they're handling all
        // memory management.
        Value * const currentAddr = b.CreateAlignedLoad(addrPtrTy, baseAddressField, voidPtrTyAlign);
        b.CreateCondBr(b.CreateICmpEQ(currentAddr, nullAddrPtr), allocBuffers, exit);

        b.SetInsertPoint(allocBuffers);

        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            b.CreateAssert(capacity, "capacity cannot be 0.");
        }

        const auto typeSize = b.getTypeSize(DL, mType);
        Rational stridesPerPage{getPageSize(), typeSize};
        capacity = b.CreateRoundUpRational(capacity, stridesPerPage.numerator());
        Value * capacityBytes = b.CreateMul(b.getSize(typeSize), capacity);

        FixedArray<Value *, 2> makeArgs;
        makeArgs[0] = capacityBytes;
        makeArgs[1] = b.getSize(0);
        Function * makeBuffer = m->getFunction(__MAKE_CIRCULAR_BUFFER); assert (makeBuffer);
        Value * baseAddress = b.CreateCall(makeBuffer, makeArgs);

        indices[1] = b.getInt32(LinearInternalCapacity);
        assert (mHandleType->getElementType(LinearInternalCapacity)->isIntegerTy());
        Value * const capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);

        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * const initialField = b.CreateInBoundsGEP(handleTy, handle, indices);

        indices[1] = b.getInt32(LinearEffectiveCapacity);
        Value * const effCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);

        b.CreateStore(baseAddress, baseAddressField);
        b.CreateStore(capacity, capacityField);
        b.CreateStore(baseAddress, initialField);
        b.CreateStore(capacity, effCapacityField);

        b.CreateBr(exit);

        b.SetInsertPoint(exit);

        b.CreateRetVoid();

        b.restoreIP(ip);
    }

    FixedArray<Value *, 2> args;
    args[0] = getHandle();
    args[1] = capacityMultiplier;
    b.CreateCall(f, args);
}

void ManagedDynamicBuffer::releaseBuffer(KernelBuilder & b) const {
    /* Free the dynamically allocated buffer(s). */
    Value * const handle = getHandle();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(LinearMallocedAddress);
    Value * const baseAddressField = b.CreateInBoundsGEP(mHandleType, handle, indices);
    Value * const baseAddress = b.CreateLoad(getPointerType(), baseAddressField);
    indices[1] = b.getInt32(LinearInternalCapacity);
    Value * const capacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    Value * const capacity = b.CreateLoad(b.getSizeTy(), capacityField);
    destroyBuffer(b, baseAddress, capacity);
    b.CreateStore(ConstantPointerNull::get(getPointerType()), baseAddressField);
}


void ManagedDynamicBuffer::destroyBuffer(KernelBuilder & b, Value * baseAddress, Value * capacity) const {
    FixedArray<Value *, 3> destroyArgs;
    destroyArgs[0] = b.CreatePointerCast(baseAddress, b.getInt8PtrTy());
    destroyArgs[1] = b.CreateMul(b.getTypeSize(mType), capacity);
    destroyArgs[2] = b.getSize(0);
    b.CreateCall(b.getModule()->getFunction(__DESTROY_CIRCULAR_BUFFER), destroyArgs);
}

Value * ManagedDynamicBuffer::getBaseAddress(KernelBuilder & b) const {
    assert (getHandle());
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * ptr = nullptr;
    indices[1] = b.getInt32(LinearBaseAddress);
    ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    return b.CreateLoad(getPointerType(), ptr);
}

Value * ManagedDynamicBuffer::getMallocAddress(KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(LinearMallocedAddress);
    Value * ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    return b.CreateLoad(getPointerType(), ptr);
}

Value * ManagedDynamicBuffer::getCapacity(KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(LinearEffectiveCapacity);
    Value * ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    Value * const capacity = b.CreateLoad(b.getSizeTy(), ptr);
    assert (capacity->getType()->isIntegerTy());
    return b.CreateMul(capacity, BLOCK_WIDTH, "capacity");
}

Value * ManagedDynamicBuffer::getInternalCapacity(KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(LinearInternalCapacity);
    Value * const intCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    Value * const capacity = b.CreateLoad(b.getSizeTy(), intCapacityField);
    assert (capacity->getType()->isIntegerTy());
    return b.CreateMul(capacity, BLOCK_WIDTH, "internalCapacity");
}


void ManagedDynamicBuffer::setBaseAddress(KernelBuilder & b, Value * const addr) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(LinearBaseAddress);
    Value * ptr = b.CreatePointerBitCastOrAddrSpaceCast(addr, getPointerType());
    b.CreateStore(ptr, b.CreateInBoundsGEP(mHandleType, mHandle, indices));
}

void ManagedDynamicBuffer::setCapacity(KernelBuilder & b, Value * const capacity) const {
    assert (mHandle && "has not been set prior to calling setCapacity");
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(LinearEffectiveCapacity);
    b.CreateStore(capacity, b.CreateInBoundsGEP(mHandleType, mHandle, indices));
}

Value * ManagedDynamicBuffer::modByCapacity(KernelBuilder & b, Value * const offset) const {
    return offset;
}

Value * ManagedDynamicBuffer::requiresExpansion(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    unsupported("requiresExpansion", "ManagedDynamicBuffer");
}

void ManagedDynamicBuffer::linearCopyBack(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    unsupported("linearCopyBack", "ManagedDynamicBuffer");
}

StructType * ManagedDynamicBuffer::getInternalThreadLocalHandleType(KernelBuilder & b) {
    FixedArray<Type *, 3> fields;
    PointerType * const voidPtrTy = b.getVoidPtrTy();
    fields[PriorAddress] = voidPtrTy;
    fields[PriorCapacity] = b.getSizeTy();
    fields[NewAddress] = voidPtrTy;
    return StructType::get(b.getContext(), fields);
}

void ManagedDynamicBuffer::freePendingDeletion(KernelBuilder & b, Value * threadLocalHandle) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(PriorAddress);
    StructType * const threadLocalTy = getInternalThreadLocalHandleType(b);
    Value * addrPtr = b.CreateGEP(threadLocalTy, threadLocalHandle, indices);
    auto & dl = b.getModule()->getDataLayout();
    PointerType * const voidPtrTy = b.getVoidPtrTy();
    IntegerType * const sizeTy = b.getSizeTy();
    FixedArray<Value *, 3> destroyArgs;
    destroyArgs[0] = b.CreateAlignedLoad(voidPtrTy, addrPtr, dl.getABITypeAlign(voidPtrTy).value());
    indices[1] = b.getInt32(PriorCapacity);
    Value * ptr = b.CreateInBoundsGEP(threadLocalTy, threadLocalHandle, indices);
    Value * capacity = b.CreateAlignedLoad(sizeTy, ptr, dl.getABITypeAlign(sizeTy).value());
    destroyBuffer(b, ptr, capacity);
}

Value * ManagedDynamicBuffer::expandBuffer(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    assert ("unspecified module" && b.getModule());

    name << "__ManagedDynamicBuffer_reserve_capacity";

    Type * ty = getBaseType();
    const auto streamCount = ty->getArrayNumElements();
    name << streamCount << 'x';
    ty = ty->getArrayElementType();
    ty = cast<FixedVectorType>(ty)->getElementType();
    const auto itemWidth = ty->getIntegerBitWidth();
    name << itemWidth << '@' << mAddressSpace;

    assert (!mLinear);

    Module * const m = b.getModule();

    Function * f = m->getFunction(name.str());

    if (f == nullptr) {

        IntegerType * const sizeTy = b.getSizeTy();

        StructType * handleTy = getHandleType(b);
        PointerType * const handlePtrTy = handleTy->getPointerTo(mAddressSpace);

        StructType * threadLocalTy = getThreadLocalHandleType(b);
        PointerType * threadLocalPtrTy = threadLocalTy->getPointerTo(mAddressSpace);



        ConstantInt * const sz_ZERO = b.getSize(0);

        auto & DL = m->getDataLayout();

        PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);

        Constant * const nullAddrPtr = ConstantPointerNull::get(addrPtrTy);

        const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();
        const auto sizeTyAlign = DL.getABITypeAlign(sizeTy).value();

        FixedArray<Type *, 5> paramTypes;
        paramTypes[0] = handlePtrTy;
        paramTypes[1] = threadLocalPtrTy; // thread local struct ptr
        paramTypes[2] = sizeTy;
        paramTypes[3] = sizeTy;
        paramTypes[4] = sizeTy;

        FunctionType * funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

        const auto ip = b.saveIP();

        f = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            f->setHasUWTable();
            #else
            f->setUWTableKind(UWTableKind::Default);
            #endif
        } else {
            f->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);
        }

        LLVMContext & C = m->getContext();
        BasicBlock * const entry = BasicBlock::Create(C, "entry", f);
        BasicBlock * const checkPendingDeletion = BasicBlock::Create(C, "checkPendingDeletion", f);
        BasicBlock * const freeExistingBuffer = BasicBlock::Create(C, "freeExistingBuffer", f);
        BasicBlock * const expandInternalBuffer = BasicBlock::Create(C, "expandAndCopyBack", f);
        BasicBlock * const afterExpandInternalBuffer = BasicBlock::Create(C, "afterCopyBackOrExpand", f);
        BasicBlock * const exit = BasicBlock::Create(C, "exit", f);

        b.SetInsertPoint(entry);

        auto arg = f->arg_begin();
        auto nextArg = [&]() {
            assert (arg != f->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };

        Value * const handle = nextArg();
        handle->setName("handle");
        Value * const threadLocalHandle = nextArg();
        threadLocalHandle->setName("threadLocalHandle");
        Value * const produced = nextArg();
        produced->setName("produced");
        Value * const consumed = nextArg();
        consumed->setName("consumed");
        Value * const required = nextArg();
        required->setName("required");
        assert (arg == f->arg_end());

        const auto blockWidth = b.getBitBlockWidth();

        ConstantInt * const BLOCK_WIDTH = b.getSize(blockWidth);

        Value * const consumedChunks = b.CreateUDiv(consumed, BLOCK_WIDTH);
        Value * const producedChunks = b.CreateCeilUDiv(produced, BLOCK_WIDTH);
        Value * const requiredCapacity = b.CreateAdd(produced, required);
        Value * const requiredChunks = b.CreateCeilUDiv(requiredCapacity, BLOCK_WIDTH);

        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(0);

        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * const currentAddressPtr = b.CreateInBoundsGEP(handleTy, handle, indices);
        Value * currentAddress = b.CreateAlignedLoad(addrPtrTy, currentAddressPtr, voidPtrTyAlign);

        indices[1] = b.getInt32(LinearInternalCapacity);
        Value * const internalCapacityPtr = b.CreateInBoundsGEP(handleTy, handle, indices);
        Value * const initialCapacity = b.CreateAlignedLoad(sizeTy, internalCapacityPtr, sizeTyAlign);

        Value * const unread = b.CreateSub(requiredChunks, consumedChunks);
        Value * const permitted = b.CreateICmpULT(unread, initialCapacity);
        b.CreateLikelyCondBr(permitted, exit, expandInternalBuffer);

        // Otherwise, allocate a buffer with at least twice the capacity and copy the unconsumed data back into it
        b.SetInsertPoint(expandInternalBuffer);

        const auto typeSize = b.getTypeSize(DL, mType);

        Rational stridesPerPage{getPageSize(), typeSize};

        Value * expandedCapacity = b.CreateRoundUp(requiredChunks, initialCapacity);
        expandedCapacity = b.CreateRoundUpRational(expandedCapacity, stridesPerPage);

        ConstantInt * const sz_TypeSize = b.getSize(typeSize);

        Value * expandedBytes = b.CreateMul(expandedCapacity, sz_TypeSize);

        FixedArray<Value *, 2> makeArgs;
        makeArgs[0] = expandedBytes;
        makeArgs[1] = sz_ZERO;
        Function * const makeBuffer = m->getFunction(__MAKE_CIRCULAR_BUFFER); assert (makeBuffer);
        Value * const expandedBuffer = b.CreatePointerCast(b.CreateCall(makeBuffer, makeArgs), addrPtrTy);

        // copy the data over to the new buffer
        Value * const newBufferOffset = b.CreateURem(consumedChunks, expandedCapacity);
        Value * const toCopyPtr = b.CreateInBoundsGEP(mType, expandedBuffer, newBufferOffset);

        Value * const oldBufferOffset = b.CreateURem(consumedChunks, initialCapacity);
        Value * const unreadDataPtr = b.CreateInBoundsGEP(mType, currentAddress, oldBufferOffset);

        Value * const unreadChunks = b.CreateSub(producedChunks, consumedChunks);

        // TODO: make this into a duff's loop
        Value * const remainingBytes = b.CreateMul(unreadChunks, sz_TypeSize);
        b.CreateMemCpy(toCopyPtr, unreadDataPtr, remainingBytes, blockWidth / 8);

        // TODO: does this need the double buffer trick used in dynamicbuffer?

        b.CreateAlignedStore(expandedBuffer, currentAddressPtr, voidPtrTyAlign);

        b.CreateAlignedStore(expandedCapacity, internalCapacityPtr, sizeTyAlign);

        // now check if there is anything to delete
        indices[1] = b.getInt32(ThreadLocalField::PriorAddress);
        Value * const priorBufferPtr = b.CreateInBoundsGEP(threadLocalTy, threadLocalHandle, indices);
        Value * const priorBuffer = b.CreateAlignedLoad(addrPtrTy, priorBufferPtr, voidPtrTyAlign);

        indices[1] = b.getInt32(ThreadLocalField::PriorCapacity);
        Value * const priorCapacityPtr = b.CreateInBoundsGEP(threadLocalTy, threadLocalHandle, indices);
        Value * const priorCapacity = b.CreateAlignedLoad(sizeTy, priorCapacityPtr, sizeTyAlign);

        indices[1] = b.getInt32(ThreadLocalField::NewAddress);
        Value * const recentAddressPtr = b.CreateInBoundsGEP(threadLocalTy, threadLocalHandle, indices);
        Value * const recentlyCreatedUnreleasedAddress = b.CreateAlignedLoad(addrPtrTy, recentAddressPtr, voidPtrTyAlign);

        // If the user created a buffer in the same logical segment, free the old one as it hasn't been released
        // to the outer system. However, if we did just create it then the prior buffer we stored isn't safe to
        // delete yet.
        Value * const hasRecent = b.CreateICmpEQ(currentAddress, recentlyCreatedUnreleasedAddress);
        b.CreateUnlikelyCondBr(hasRecent, freeExistingBuffer, checkPendingDeletion);

        b.SetInsertPoint(checkPendingDeletion);
        // if the user hasn't created a buffer in this segment but expanded the buffer in a prior segment,
        // free it now as the pipeline has completed a full iteration
        b.CreateCondBr(b.CreateICmpNE(priorBuffer, nullAddrPtr), freeExistingBuffer, afterExpandInternalBuffer);

        b.SetInsertPoint(freeExistingBuffer);
        PHINode * const toDeleteBufferPhi = b.CreatePHI(addrPtrTy, 2);
        toDeleteBufferPhi->addIncoming(currentAddress, expandInternalBuffer);
        toDeleteBufferPhi->addIncoming(priorBuffer, checkPendingDeletion);

        PHINode * const toDeleteCapacityPhi = b.CreatePHI(sizeTy, 2);
        toDeleteCapacityPhi->addIncoming(initialCapacity, expandInternalBuffer);
        toDeleteCapacityPhi->addIncoming(priorCapacity, checkPendingDeletion);

        PHINode * const pendingDeleteBufferPhi = b.CreatePHI(addrPtrTy, 2);
        pendingDeleteBufferPhi->addIncoming(priorBuffer, expandInternalBuffer);
        pendingDeleteBufferPhi->addIncoming(nullAddrPtr, checkPendingDeletion);

        PHINode * const pendingDeleteCapacityPhi = b.CreatePHI(sizeTy, 2);
        pendingDeleteCapacityPhi->addIncoming(priorCapacity, expandInternalBuffer);
        pendingDeleteCapacityPhi->addIncoming(sz_ZERO, checkPendingDeletion);

        destroyBuffer(b, toDeleteBufferPhi, toDeleteCapacityPhi);

        b.CreateBr(afterExpandInternalBuffer);

        b.SetInsertPoint(afterExpandInternalBuffer);
        PHINode * const priorAllocAddrPhi = b.CreatePHI(addrPtrTy, 2);
        priorAllocAddrPhi->addIncoming(currentAddress, checkPendingDeletion);
        priorAllocAddrPhi->addIncoming(pendingDeleteBufferPhi, freeExistingBuffer);

        PHINode * const priorAllocCapacityPhi = b.CreatePHI(sizeTy, 2);
        priorAllocCapacityPhi->addIncoming(initialCapacity, checkPendingDeletion);
        priorAllocCapacityPhi->addIncoming(pendingDeleteCapacityPhi, freeExistingBuffer);

        b.CreateAlignedStore(priorAllocAddrPhi, priorBufferPtr, voidPtrTyAlign);
        b.CreateAlignedStore(priorAllocCapacityPhi, priorCapacityPtr, voidPtrTyAlign);
        b.CreateAlignedStore(expandedBuffer, currentAddressPtr, voidPtrTyAlign);
        b.CreateAlignedStore(expandedBuffer, recentAddressPtr, voidPtrTyAlign);
        b.CreateBr(exit);

        b.SetInsertPoint(exit);
        PHINode * const currentAllocAddrPhi = b.CreatePHI(addrPtrTy, 2);
        currentAllocAddrPhi->addIncoming(currentAddress, entry);
        currentAllocAddrPhi->addIncoming(expandedBuffer, afterExpandInternalBuffer);

        PHINode * const currentAllocCapacityPhi = b.CreatePHI(sizeTy, 2);
        currentAllocCapacityPhi->addIncoming(initialCapacity, entry);
        currentAllocCapacityPhi->addIncoming(expandedCapacity, afterExpandInternalBuffer);

        indices[1] = b.getInt32(LinearBaseAddress);
        Value * const virtualBaseField = b.CreateInBoundsGEP(handleTy, handle, indices);

        Value * consumedOffset = b.CreateSub(b.CreateURem(consumedChunks, currentAllocCapacityPhi), consumedChunks);

        Value * const newVirtualAddress = b.CreateInBoundsGEP(mType, currentAllocAddrPhi, consumedOffset);
        b.CreateAlignedStore(newVirtualAddress, virtualBaseField, voidPtrTyAlign);

        indices[1] = b.getInt32(LinearEffectiveCapacity);
        Value * const effCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
        b.CreateAlignedStore(requiredChunks, effCapacityField, sizeTyAlign);

        b.CreateRetVoid();

        b.restoreIP(ip);

    }

    FixedArray<Value *, 5> args;
    args[0] = mHandle;
    args[1] = mThreadLocalHandle;
    args[2] = produced;
    args[3] = consumed;
    args[4] = required;

    return b.CreateCall(f, args);

}


// Repeating Buffer

StructType * RepeatingBuffer::getHandleType(KernelBuilder & b) const {
    if (mHandleType == nullptr) {
        auto & C = b.getContext();
        FixedArray<Type *, 1> types;
        types[BaseAddress] = getPointerType();
        mHandleType = StructType::get(C, types);
    }
    return mHandleType;
}

void RepeatingBuffer::allocateBuffer(KernelBuilder & b, Value * const capacityMultiplier) {
    unsupported("allocateBuffer", "Repeating");
}

void RepeatingBuffer::releaseBuffer(KernelBuilder & b) const {
    unsupported("releaseBuffer", "Repeating");
}

void RepeatingBuffer::destroyBuffer(KernelBuilder & b, Value * baseAddress, Value *capacity) const {

}

Value * RepeatingBuffer::modByCapacity(KernelBuilder & b, Value * const offset) const {
    Value * const capacity = b.CreateExactUDiv(mModulus, b.getSize(b.getBitBlockWidth()));
    return b.CreateURem(offset, capacity);
}

Value * RepeatingBuffer::getCapacity(KernelBuilder & b) const {
    return mModulus;
}

Value * RepeatingBuffer::getInternalCapacity(KernelBuilder & b) const {
    return mModulus;
}

void RepeatingBuffer::setCapacity(KernelBuilder & b, Value * capacity) const {
    unsupported("setCapacity", "Repeating");
}


Value * RepeatingBuffer::getVirtualBasePtr(KernelBuilder & b, Value * const baseAddress, Value * const transferredItems) const {
    Value * addr = nullptr;
    Constant * const LOG_2_BLOCK_WIDTH = b.getSize(floor_log2(b.getBitBlockWidth()));
    if (mUnaligned) {
        assert (isConstantOne(getStreamSetCount(b)));
        Value * offset = b.CreateSub(transferredItems, b.CreateURem(transferredItems, mModulus));
        Type * const elemTy = cast<ArrayType>(mBaseType)->getElementType();
        Type * itemTy = cast<VectorType>(elemTy)->getElementType();
        #if LLVM_VERSION_CODE < LLVM_VERSION_CODE(12, 0, 0)
        const unsigned itemWidth = itemTy->getPrimitiveSizeInBits();
        #else
        const unsigned itemWidth = itemTy->getPrimitiveSizeInBits().getFixedSize();
        #endif
        PointerType * itemPtrTy = nullptr;
        if (LLVM_UNLIKELY(itemWidth < 8)) {
            const Rational itemsPerByte{8, itemWidth};
            offset = b.CreateUDivRational(offset, itemsPerByte);
            itemTy = b.getInt8Ty();
        }
        itemPtrTy = itemTy->getPointerTo(mAddressSpace);
        addr = b.CreatePointerCast(baseAddress, itemPtrTy);
        addr = b.CreateInBoundsGEP(itemTy, addr, b.CreateNeg(offset));
    } else {
        Value * const transferredBlocks = b.CreateLShr(transferredItems, LOG_2_BLOCK_WIDTH);
        Constant * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
        Value * const capacity = b.CreateExactUDiv(mModulus, BLOCK_WIDTH);
        Value * offset = b.CreateURem(transferredBlocks, capacity);
        offset = b.CreateSub(offset, transferredBlocks);
        Constant * const sz_ZERO = b.getSize(0);
        addr = StreamSetBuffer::getStreamBlockPtr(b, baseAddress, sz_ZERO, offset);
    }
    return b.CreatePointerCast(addr, getPointerType());
}

Value * RepeatingBuffer::getBaseAddress(KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    Value * const base = b.CreateInBoundsGEP(mHandleType, handle, indices);
    return b.CreateLoad(getPointerType(), base, "baseAddress");
}

void RepeatingBuffer::setBaseAddress(KernelBuilder & b, Value * addr) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    b.CreateStore(addr, b.CreateInBoundsGEP(mHandleType, handle, indices));
}

Value * RepeatingBuffer::getMallocAddress(KernelBuilder & b) const {
    return getBaseAddress(b);
}

Value * RepeatingBuffer::requiresExpansion(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    return b.getFalse();
}

void RepeatingBuffer::linearCopyBack(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    unsupported("linearCopyBack", "Repeating");
}

Value * RepeatingBuffer::expandBuffer(KernelBuilder & b, Value * produced, Value * consumed, Value * const required) const  {
    unsupported("linearCopyBack", "Repeating");
}

// Constructors

ExternalBuffer::ExternalBuffer(const unsigned id, KernelBuilder & b, Type * const type,
                               const unsigned AddressSpace)
: StreamSetBuffer(id, BufferKind::ExternalBuffer, b, type,  true, AddressSpace) {

}

DynamicBuffer::DynamicBuffer(const unsigned id, kernel::KernelBuilder & b, Type * const type,
                             const bool hasUnderflow,
                             const bool linear, const unsigned AddressSpace)
: InternalBuffer(id, BufferKind::DynamicBuffer, b, type, linear, AddressSpace)
, mHasUnderflow(hasUnderflow) {

}

ManagedDynamicBuffer::ManagedDynamicBuffer(const unsigned id, kernel::KernelBuilder & b,
                                           llvm::Type * const type, const unsigned AddressSpace)
: InternalBuffer(id, BufferKind::ManagedDynamicBuffer, b, type, false, AddressSpace) {

}

RepeatingBuffer::RepeatingBuffer(const unsigned id, KernelBuilder & b, Type * const type, const bool unaligned)
: InternalBuffer(id, BufferKind::RepeatingBuffer, b, type, false, 0)
, mUnaligned(unaligned) {

}


inline InternalBuffer::InternalBuffer(const unsigned id, const BufferKind k, KernelBuilder & b, Type * const baseType,
                                      const bool linear, const unsigned AddressSpace)
: StreamSetBuffer(id, k, b, baseType, linear, AddressSpace) {

}

inline StreamSetBuffer::StreamSetBuffer(const unsigned id, const BufferKind k, KernelBuilder & b, Type * const baseType,
                                        const bool linear, const unsigned AddressSpace)
: mId(id)
, mBufferKind(k)
, mHandle(nullptr)
, mType(resolveType(b, baseType))
, mBaseType(baseType)
, mHandleType(nullptr)
, mAddressSpace(AddressSpace)
, mLinear(linear || isEmptySet()) {

}

StreamSetBuffer::~StreamSetBuffer() { }

}
