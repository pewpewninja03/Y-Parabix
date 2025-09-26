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
#include <llvm/IR/GlobalVariable.h>

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

#include <llvm/Support/raw_os_ostream.h>

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
    llvm::errs() << StringRef{"NEW CIRCULAR BUFFER "}; llvm::errs().write_hex((uintptr_t)base) << " -> ";
    llvm::errs().write_hex(((uintptr_t)base) + size) << '\n';
    return base + (hasUnderflow * size);
}

void destroy_circular_buffer(uint8_t * base, const size_t size, const size_t hasUnderflow) {
    if (LLVM_UNLIKELY(base == nullptr)) {
        return;
    }
    const size_t n = hasUnderflow ? 1 : 0;
    const size_t m = n + 2;
    const auto ptr = base - (n * size);
    llvm::errs() << StringRef{"DELETE CIRCULAR BUFFER "}; llvm::errs().write_hex((uintptr_t)ptr) << " -> ";
    llvm::errs().write_hex(((uintptr_t)base) + m * size) << '\n';
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

Value * ExternalBuffer::reserveCapacity(KernelBuilder & /* b */, Value * /* produced */, Value * /* consumed */, Value * const /* required */) const  {
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
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts, codegen::EnableStreamSetAsserts))) {
            Value * const valid = b.CreateICmpULE(producedItems, capacity);
            b.CreateAssert(valid, "produced item count (%" PRIu64 ") exceeds capacity (%" PRIu64 ").",
                            producedItems, capacity);
        }
        return b.CreateSub(capacity, producedItems);
     } else {
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts, codegen::EnableStreamSetAsserts))) {
            Value * const valid = b.CreateICmpULE(consumedItems, producedItems);
            b.CreateAssert(valid, "consumed item count (%" PRIu64 ") exceeds produced (%" PRIu64 ").",
                            consumedItems, producedItems);
        }
        Value * const unconsumedItems = b.CreateSub(producedItems, consumedItems);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts, codegen::EnableStreamSetAsserts))) {
            Value * const valid = b.CreateICmpULE(unconsumedItems, capacity);
            b.CreateAssert(valid, "unconsumed item count (%" PRIu64 ") exceeds capacity (%" PRIu64 ").",
                            unconsumedItems, capacity);
        }
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

Value * DynamicBuffer::reserveCapacity(KernelBuilder & b, Value * const produced, Value * const consumed, Value * const required) const {

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

        b.CallPrintInt(name.str(), handle);

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
        IntegerType * const sizeTy = b.getSizeTy();
        PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);
        StructType * const emptTy = StructType::get(C);
        FixedArray<Type *, 7> fields;
        fields[LinearMallocedAddress] = addrPtrTy;
        fields[LinearInternalCapacity] = sizeTy;
        if (mLinear) {
            fields[SecondLinearMallocedAddress] = emptTy;
            fields[SecondLinearInternalCapacity] = emptTy;
            fields[LinearSelector] = emptTy;
            fields[LinearBaseAddress] = addrPtrTy;
            fields[LinearEffectiveCapacity] = sizeTy;
        } else {
            fields[SecondLinearMallocedAddress] = addrPtrTy;
            fields[SecondLinearInternalCapacity] = sizeTy;
            fields[LinearSelector] = sizeTy;
            fields[LinearBaseAddress] = emptTy;
            fields[LinearEffectiveCapacity] = emptTy;
        }
        mHandleType = StructType::get(C, fields);
    }
    return mHandleType;
}

Value * ManagedDynamicBuffer::getVirtualBasePtr(KernelBuilder & b, Value * const baseAddress, Value * const transferredItems) const {
    Value * addr = baseAddress;
    if (!mLinear) {
        Constant * const LOG_2_BLOCK_WIDTH = b.getSize(floor_log2(b.getBitBlockWidth()));
        Value * const blockIndex = b.CreateLShr(transferredItems, LOG_2_BLOCK_WIDTH);
        Value * const baseBlockIndex = b.CreateSub(modByCapacity(b, blockIndex), blockIndex);
        addr = StreamSetBuffer::getStreamBlockPtr(b, baseAddress, b.getSize(0), baseBlockIndex);
    }
    return b.CreatePointerCast(addr, getPointerType());
}

void ManagedDynamicBuffer::allocateBuffer(KernelBuilder & b, Value * const capacityMultiplier) {
    assert (mHandle && "has not been set prior to calling allocateBuffer");

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    assert ("unspecified module" && b.getModule());

    name << "__ManagedDynamicBuffer";
    if (mLinear) {
        name << "Linear";
    }
    name << "_initial_alloc";

    Type * ty = getBaseType();
    const auto streamCount = ty->getArrayNumElements();
    name << streamCount << 'x';
    ty = ty->getArrayElementType();
    ty = cast<FixedVectorType>(ty)->getElementType();
    const auto itemWidth = ty->getIntegerBitWidth();
    name << itemWidth << '@' << mAddressSpace;

    Module * const m = b.getModule();

    Function * f = m->getFunction(name.str());

    if (f == nullptr) {

        StructType * handleTy = getHandleType(b);
        PointerType * const handlePtrTy = handleTy->getPointerTo(mAddressSpace);

        auto & DL = m->getDataLayout();

        PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);
        IntegerType * const intPtrTy = b.getIntPtrTy(DL);

        const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();
        const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

        FixedArray<Type *, 2> paramTypes;
        paramTypes[0] = handlePtrTy;
        paramTypes[1] = intPtrTy;

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
        BasicBlock * allocBuffers = nullptr;
        BasicBlock * exit = nullptr;

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

        Constant * nullVoidPtr = ConstantPointerNull::get(addrPtrTy);

        Value * baseAddressField = nullptr;
        if (mLinear) {

            allocBuffers = BasicBlock::Create(C, "allocBuffers", f);
            exit = BasicBlock::Create(C, "exit", f);
            indices[1] = b.getInt32(LinearBaseAddress);

            baseAddressField = b.CreateInBoundsGEP(handleTy, handle, indices);
            Value * currentAddr = b.CreateAlignedLoad(addrPtrTy, baseAddressField, voidPtrTyAlign);
            // If the user has filled in a base address in the init function, assume they're handling all
            // memory management.
            b.CreateCondBr(b.CreateICmpEQ(currentAddr, nullVoidPtr), allocBuffers, exit);

            b.SetInsertPoint(allocBuffers);
        }

        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            b.CreateAssert(capacity, "capacity cannot be 0.");
        }

        const auto typeSize = b.getTypeSize(DL, mType);
        Rational stridesPerPage{getPageSize(), typeSize};
        capacity = b.CreateRoundUpRational(capacity, stridesPerPage.numerator());
        Value * capacityBytes = b.CreateMul(ConstantInt::get(intPtrTy, typeSize), capacity);

        ConstantInt * const sz_ZERO = b.getSize(0);

        FixedArray<Value *, 2> makeArgs;
        makeArgs[0] = capacityBytes;
        makeArgs[1] = sz_ZERO;
        Function * makeBuffer = m->getFunction(__MAKE_CIRCULAR_BUFFER); assert (makeBuffer);
        Value * baseAddress = b.CreateCall(makeBuffer, makeArgs);

        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * mallocAddrField = b.CreateInBoundsGEP(handleTy, handle, indices);
        b.CreateAlignedStore(baseAddress, mallocAddrField, voidPtrTyAlign);

        indices[1] = b.getInt32(LinearInternalCapacity);
        Value * capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
        b.CreateAlignedStore(capacity, capacityField, intPtrTyAlign);

        if (mLinear) {

            b.CreateAlignedStore(baseAddress, baseAddressField, voidPtrTyAlign);

            indices[1] = b.getInt32(LinearEffectiveCapacity);
            Value * effCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
            b.CreateAlignedStore(capacity, effCapacityField, intPtrTyAlign);

            b.CreateBr(exit);

            b.SetInsertPoint(exit);

        } else {

            indices[1] = b.getInt32(SecondLinearMallocedAddress);
            Value * toWriteSecondAddrField = b.CreateInBoundsGEP(handleTy, handle, indices);
            b.CreateAlignedStore(baseAddress, toWriteSecondAddrField, voidPtrTyAlign);

            indices[1] = b.getInt32(SecondLinearInternalCapacity);
            Value * toWriteSecondCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
            b.CreateAlignedStore(capacity, toWriteSecondCapacityField, intPtrTyAlign);

            indices[1] = b.getInt32(LinearSelector);
            Value * linearSelectorField = b.CreateInBoundsGEP(handleTy, handle, indices);
            b.CreateAlignedStore(sz_ZERO, linearSelectorField, intPtrTyAlign);

        }
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

    StructType * handleTy = getHandleType(b);

    Module * m = b.getModule();

    auto & DL = m->getDataLayout();

    PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);
    IntegerType * const intPtrTy = b.getIntPtrTy(DL);

    const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();
    const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

    Value * const handle = getHandle();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);

    Value * addrField = nullptr;
    Value * capacityField = nullptr;

    if (mLinear) {
        indices[1] = b.getInt32(LinearMallocedAddress);
        addrField = b.CreateInBoundsGEP(handleTy, handle, indices);

        indices[1] = b.getInt32(LinearInternalCapacity);
        capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);

    } else {

        indices[1] = b.getInt32(LinearSelector);
        Value * linearSelectorField = b.CreateInBoundsGEP(handleTy, handle, indices);
        Value * const selector = b.CreateAlignedLoad(intPtrTy, linearSelectorField, intPtrTyAlign);
        Value * const bSelector = b.CreateICmpEQ(selector, b.getSize(0));

        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * primaryAddr = b.CreateInBoundsGEP(handleTy, handle, indices);
        indices[1] = b.getInt32(SecondLinearMallocedAddress);
        Value * secondaryAddr = b.CreateInBoundsGEP(handleTy, handle, indices);


        indices[1] = b.getInt32(LinearInternalCapacity);
        Value * primaryCap = b.CreateInBoundsGEP(handleTy, handle, indices);
        indices[1] = b.getInt32(SecondLinearInternalCapacity);
        Value * secondaryCap = b.CreateInBoundsGEP(handleTy, handle, indices);


        addrField = b.CreateSelect(bSelector, primaryAddr, secondaryAddr);

        capacityField = b.CreateSelect(bSelector, primaryCap, secondaryCap);
    }

    Value * addr = b.CreateAlignedLoad(addrPtrTy, addrField, voidPtrTyAlign);
    Value * capacity = b.CreateAlignedLoad(intPtrTy, capacityField, intPtrTyAlign);

    destroyBuffer(b, addr, capacity);

    b.CreateAlignedStore(ConstantPointerNull::get(addrPtrTy), addrField, voidPtrTyAlign);
}


void ManagedDynamicBuffer::destroyBuffer(KernelBuilder & b, Value * baseAddress, Value * capacity) const {
    FixedArray<Value *, 3> destroyArgs;
    destroyArgs[0] = b.CreatePointerCast(baseAddress, b.getInt8PtrTy());
    b.CallPrintInt("destroyBuffer::capacity", capacity);
    destroyArgs[1] = b.CreateMul(b.getTypeSize(mType), capacity);
    b.CallPrintInt("destroyBuffer::size", destroyArgs[1]);
    destroyArgs[2] = b.getSize(0);
    b.CreateCall(b.getModule()->getFunction(__DESTROY_CIRCULAR_BUFFER), destroyArgs);
}

Value * ManagedDynamicBuffer::getBaseAddress(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    auto & DL = b.getModule()->getDataLayout();
    IntegerType * const intPtrTy = b.getIntPtrTy(DL);
    const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();
    PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);
    const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * field = nullptr;
    if (mLinear) {
        indices[1] = b.getInt32(LinearBaseAddress);
        field = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    } else {
        indices[1] = b.getInt32(LinearSelector);
        Value * selectorField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * selector = b.CreateAlignedLoad(intPtrTy, selectorField, intPtrTyAlign);
        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * primaryAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(SecondLinearMallocedAddress);
        Value * secondaryAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        field = b.CreateSelect(b.CreateICmpEQ(selector, b.getSize(0)), primaryAddr, secondaryAddr);
    }
    return b.CreateAlignedLoad(addrPtrTy, field, voidPtrTyAlign);
}

Value * ManagedDynamicBuffer::getMallocAddress(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    auto & DL = b.getModule()->getDataLayout();
    IntegerType * const intPtrTy = b.getIntPtrTy(DL);
    const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();
    PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);
    const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * field = nullptr;
    if (mLinear) {
        indices[1] = b.getInt32(LinearMallocedAddress);
        field = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    } else {
        indices[1] = b.getInt32(LinearSelector);
        Value * selectorField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * selector = b.CreateAlignedLoad(intPtrTy, selectorField, intPtrTyAlign);
        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * primaryAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(SecondLinearMallocedAddress);
        Value * secondaryAddr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        field = b.CreateSelect(b.CreateICmpEQ(selector, b.getSize(0)), primaryAddr, secondaryAddr);
    }
    return b.CreateAlignedLoad(addrPtrTy, field, voidPtrTyAlign);
}

Value * ManagedDynamicBuffer::getCapacity(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    auto & DL = b.getModule()->getDataLayout();
    IntegerType * const intPtrTy = b.getIntPtrTy(DL);
    const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * capField = nullptr;
    if (mLinear) {
        indices[1] = b.getInt32(LinearEffectiveCapacity);
        capField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    } else {
        indices[1] = b.getInt32(LinearSelector);
        Value * selectorField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * selector = b.CreateAlignedLoad(intPtrTy, selectorField, intPtrTyAlign);
        indices[1] = b.getInt32(LinearInternalCapacity);
        Value * primaryCapField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(SecondLinearInternalCapacity);
        Value * secondaryCapField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        capField = b.CreateSelect(b.CreateICmpEQ(selector, b.getSize(0)), primaryCapField, secondaryCapField);
    }
    Value * cap = b.CreateAlignedLoad(intPtrTy, capField, intPtrTyAlign);
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    return b.CreateMul(BLOCK_WIDTH, cap);
}

Value * ManagedDynamicBuffer::getInternalCapacity(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    auto & DL = b.getModule()->getDataLayout();
    IntegerType * const intPtrTy = b.getIntPtrTy(DL);
    const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * capField = nullptr;
    if (mLinear) {
        indices[1] = b.getInt32(LinearInternalCapacity);
        capField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    } else {
        indices[1] = b.getInt32(LinearSelector);
        Value * selectorField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * selector = b.CreateAlignedLoad(intPtrTy, selectorField, intPtrTyAlign);
        indices[1] = b.getInt32(LinearInternalCapacity);
        Value * primaryCapField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(SecondLinearInternalCapacity);
        Value * secondaryCapField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        capField = b.CreateSelect(b.CreateICmpEQ(selector, b.getSize(0)), primaryCapField, secondaryCapField);
    }
    Value * cap = b.CreateAlignedLoad(intPtrTy, capField, intPtrTyAlign);
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    return b.CreateMul(BLOCK_WIDTH, cap);
}

void ManagedDynamicBuffer::setBaseAddress(KernelBuilder & b, Value * const addr) const {
    assert (mLinear);
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    auto & DL = b.getModule()->getDataLayout();

    PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);
    const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(LinearBaseAddress);
    Value * ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    b.CreateAlignedStore(addr, ptr, voidPtrTyAlign);
}

void ManagedDynamicBuffer::setCapacity(KernelBuilder & b, Value * const capacity) const {
    assert (mLinear);
    assert (mHandle && "has not been set prior to calling setCapacity");
    auto & DL = b.getModule()->getDataLayout();
    IntegerType * const intPtrTy = b.getIntPtrTy(DL);
    const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(LinearEffectiveCapacity);
    Value * ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    b.CreateAlignedStore(capacity, ptr, intPtrTyAlign);
}

Value * ManagedDynamicBuffer::modByCapacity(KernelBuilder & b, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    if (mLinear) {
        return offset;
    } else {
        assert (mHandle && "has not been set prior to calling setBaseAddress");
        auto & DL = b.getModule()->getDataLayout();
        IntegerType * const intPtrTy = b.getIntPtrTy(DL);
        const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();
        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(0);
        indices[1] = b.getInt32(LinearSelector);
        Value * selectorField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * selector = b.CreateAlignedLoad(intPtrTy, selectorField, intPtrTyAlign);
        indices[1] = b.getInt32(LinearInternalCapacity);
        Value * primaryCap = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        indices[1] = b.getInt32(SecondLinearInternalCapacity);
        Value * secondaryCap = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * field = field = b.CreateSelect(b.CreateICmpEQ(selector, b.getSize(0)), primaryCap, secondaryCap);
        Value * cap = b.CreateAlignedLoad(intPtrTy, field, intPtrTyAlign);
        return b.CreateURem(offset, cap);
    }
}

Value * ManagedDynamicBuffer::requiresExpansion(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    unsupported("requiresExpansion", "ManagedDynamicBuffer");
}

void ManagedDynamicBuffer::linearCopyBack(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    unsupported("linearCopyBack", "ManagedDynamicBuffer");
}

StructType * ManagedDynamicBuffer::getInternalThreadLocalHandleType(KernelBuilder & b) {
    FixedArray<Type *, 3> fields;
    PointerType * const addrPtrTy = b.getInt8PtrTy();
    fields[PriorAddress] = addrPtrTy;
    fields[PriorCapacity] = b.getSizeTy();
    fields[NewAddress] = addrPtrTy;
    return StructType::get(b.getContext(), fields);
}

void ManagedDynamicBuffer::freePendingDeletion(KernelBuilder & b, Value * threadLocalHandle) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(PriorAddress);
    StructType * const threadLocalTy = getInternalThreadLocalHandleType(b);
    Value * addrPtr = b.CreateGEP(threadLocalTy, threadLocalHandle, indices);
    auto & dl = b.getModule()->getDataLayout();
    PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);
    IntegerType * const sizeTy = b.getSizeTy();
    FixedArray<Value *, 3> destroyArgs;
    destroyArgs[0] = b.CreateAlignedLoad(addrPtrTy, addrPtr, dl.getABITypeAlign(addrPtrTy).value());
    indices[1] = b.getInt32(PriorCapacity);
    Value * ptr = b.CreateInBoundsGEP(threadLocalTy, threadLocalHandle, indices);
    Value * capacity = b.CreateAlignedLoad(sizeTy, ptr, dl.getABITypeAlign(sizeTy).value());
    destroyBuffer(b, ptr, capacity);
}

Value * ManagedDynamicBuffer::reserveCapacity(KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {

    // TODO: if we keep the buffer fd, manual indicates we can resize it by calling ftruncate again. How would this
    // affect users of the buffer prior to setting the new addr/capacity? The mmap'ed buffer pointing to the fds
    // have a set size that may mean its a non-issue as long as ftruncating the buffer doesn't invalidate them.

    Module * const m = b.getModule();
    auto & DL = m->getDataLayout();

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    assert ("unspecified module" && b.getModule());

    name << "__ManagedDynamicBuffer";
    if (mLinear) {
        name << "Linear";
    }
    name << "_reserve_capacity" << mAddressSpace;

    PointerType * const voidPtrTy = b.getVoidPtrTy();
    const auto blockWidth = b.getBitBlockWidth();

    Function * f = m->getFunction(name.str());

    ConstantInt * const sz_ONE = b.getSize(1);

    if (f == nullptr) {

        const auto ip = b.saveIP();

        StructType * const handleTy = getHandleType(b);
        PointerType * const handlePtrTy = handleTy->getPointerTo(mAddressSpace);

        StructType * const threadLocalTy = getThreadLocalHandleType(b);
        PointerType * const threadLocalPtrTy = threadLocalTy->getPointerTo(mAddressSpace);

        PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);

        PointerType * const i8PtrTy = b.getInt8PtrTy();
        Constant * const nullAddrPtr = ConstantPointerNull::get(i8PtrTy);
        const auto voidPtrTyAlign = DL.getABITypeAlign(i8PtrTy).value();

        auto & C = b.getContext();
        IntegerType * const intPtrTy = DL.getIntPtrType(C);
        const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

        constexpr auto DUFF_STEPS = 8;

        FixedArray<Type *, 6> paramTypes;
        paramTypes[0] = voidPtrTy; // shared struct ptr
        paramTypes[1] = voidPtrTy; // thread local struct ptr
        paramTypes[2] = intPtrTy;
        paramTypes[3] = intPtrTy;
        paramTypes[4] = intPtrTy;
        paramTypes[5] = intPtrTy;

        Type * const retValTy = mLinear ? (Type*)intPtrTy : b.getVoidTy();

        FunctionType * funcTy = FunctionType::get(retValTy, paramTypes, false);

        f = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            f->setHasUWTable();
            #else
            f->setUWTableKind(UWTableKind::Default);
            #endif
        }

        BasicBlock * const entry = BasicBlock::Create(C, "entry", f);
        BasicBlock * const expandInternalBuffer = BasicBlock::Create(C, "expandAndCopyBack", f);
        BasicBlock * const copyEntry = BasicBlock::Create(C, "copyEntry", f);
        FixedArray<BasicBlock *, DUFF_STEPS> copyLoopEntryPoint;
        if (LLVM_LIKELY(b.supportsIndirectBr())) {
            for (unsigned i = 0; i < DUFF_STEPS; ++i) {
                copyLoopEntryPoint[i] = BasicBlock::Create(C, "copyLoopEntryPoint", f);
            }
        }
        BasicBlock * const copyExit = BasicBlock::Create(C, "copyExit", f);
        BasicBlock * const freeExistingBuffer = BasicBlock::Create(C, "freeExistingBuffer", f);
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

        Value * const handle = b.CreatePointerCast(nextArg(), handlePtrTy);
        handle->setName("handle");
        Value * const threadLocalHandle = b.CreatePointerCast(nextArg(), threadLocalPtrTy);
        threadLocalHandle->setName("threadLocalHandle");
        Value * const produced = nextArg();
        produced->setName("produced");
        Value * const consumed = nextArg();
        consumed->setName("consumed");
        Value * const required = nextArg();
        required->setName("required");
        Value * const blocksPerChunk = nextArg();
        blocksPerChunk->setName("blocksPerChunk");
        assert (arg == f->arg_end());

        b.CallPrintInt(name.str(), handle);

        ConstantInt * const BLOCK_WIDTH = b.getSize(blockWidth);
        ConstantInt * const sz_ZERO = b.getSize(0);


        Value * const consumedChunks = b.CreateUDiv(consumed, BLOCK_WIDTH);
        Value * const requiredChunks = b.CreateCeilUDiv(b.CreateAdd(produced, required), BLOCK_WIDTH);

        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(0);

        Value * selectorField = nullptr;
        Value * selector = nullptr;
        Value * addrField = nullptr;
        Value * capacityField = nullptr;
        Value * primaryAddrField = nullptr;
        Value * primaryCapacityField = nullptr;
        Value * secondaryAddrField = nullptr;
        Value * secondaryCapacityField = nullptr;

        if (mLinear) {
            indices[1] = b.getInt32(LinearMallocedAddress);
            addrField = b.CreateInBoundsGEP(handleTy, handle, indices);
            indices[1] = b.getInt32(LinearInternalCapacity);
            capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
        } else {
            indices[1] = b.getInt32(LinearSelector);
            selectorField = b.CreateInBoundsGEP(handleTy, handle, indices);
            selector = b.CreateAlignedLoad(intPtrTy, selectorField, intPtrTyAlign);
            indices[1] = b.getInt32(LinearMallocedAddress);
            primaryAddrField = b.CreateInBoundsGEP(handleTy, handle, indices);
            indices[1] = b.getInt32(LinearInternalCapacity);
            primaryCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
            indices[1] = b.getInt32(SecondLinearMallocedAddress);
            secondaryAddrField = b.CreateInBoundsGEP(handleTy, handle, indices);
            indices[1] = b.getInt32(SecondLinearInternalCapacity);
            secondaryCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
            Value * const bSelector = b.CreateICmpEQ(selector, sz_ZERO);
            addrField = b.CreateSelect(bSelector, primaryAddrField, secondaryAddrField);
            capacityField = b.CreateSelect(bSelector, primaryCapacityField, secondaryCapacityField);
        }

        Value * const initialAddr = b.CreateAlignedLoad(i8PtrTy, addrField, voidPtrTyAlign);
        Value * const initialCapacity = b.CreateAlignedLoad(intPtrTy, capacityField, intPtrTyAlign);

        assert (blockWidth > 8);

        Constant * const log2BytesPerBlock = b.getSize(floor_log2(blockWidth) - 3);

        Value * const bytesPerChunk = b.CreateShl(blocksPerChunk, log2BytesPerBlock);

        BasicBlock * reuseExistingBuffer = nullptr;

        if (mLinear) {
            reuseExistingBuffer = BasicBlock::Create(C, "reuseExistingBuffer", f, copyEntry);
            Value * const canReuse = b.CreateICmpULE(requiredChunks, consumedChunks);
            b.CreateLikelyCondBr(canReuse, reuseExistingBuffer, expandInternalBuffer);
        } else {
            b.CreateBr(expandInternalBuffer);
        }


        // Otherwise, allocate a buffer with at least twice the capacity and copy the unconsumed data back into it
        b.SetInsertPoint(expandInternalBuffer);
        Value * const pendingChunks = b.CreateSub(requiredChunks, consumedChunks);
        Value * expandedCapacity = b.CreateAdd(initialCapacity, b.CreateRoundUp(pendingChunks, initialCapacity));

        Value * const expandedBytes = b.CreateMul(expandedCapacity, bytesPerChunk);

        FixedArray<Value *, 2> makeArgs;
        makeArgs[0] = expandedBytes;
        makeArgs[1] = sz_ZERO;
        Function * const makeBuffer = m->getFunction(__MAKE_CIRCULAR_BUFFER); assert (makeBuffer);
        Value * expandedAddr = b.CreatePointerCast(b.CreateCall(makeBuffer, makeArgs), i8PtrTy);

        Value * expandedBufferRetVal = nullptr;
        if (mLinear) {
            b.CreateBr(reuseExistingBuffer);

            b.SetInsertPoint(reuseExistingBuffer);
            PHINode * const addrPhi = b.CreatePHI(i8PtrTy, 2);
            addrPhi->addIncoming(initialAddr, entry);
            addrPhi->addIncoming(expandedAddr, expandInternalBuffer);
            expandedAddr = addrPhi;
            PHINode * const capacityPhi = b.CreatePHI(intPtrTy, 2);
            capacityPhi->addIncoming(initialCapacity, entry);
            capacityPhi->addIncoming(expandedCapacity, expandInternalBuffer);
            expandedCapacity = capacityPhi;
            // TODO: return 2 if we "reuse" the data but do not copy anything?
            PHINode * const expandedBufferRetValPhi = b.CreatePHI(intPtrTy, 2);
            expandedBufferRetValPhi->addIncoming(sz_ZERO, entry);
            expandedBufferRetValPhi->addIncoming(sz_ONE, expandInternalBuffer);
            expandedBufferRetVal = expandedBufferRetValPhi;
        }


        // copy the data over to the new buffer
        Value * const producedChunks = b.CreateCeilUDiv(produced, BLOCK_WIDTH);
        Value * const unreadChunks = b.CreateSub(producedChunks, consumedChunks);

        b.CreateUnlikelyCondBr(b.CreateICmpEQ(unreadChunks, sz_ZERO), copyExit, copyEntry);

        b.SetInsertPoint(copyEntry);
        Value * toCopyPtr = nullptr;
        Value * unreadDataPtr = nullptr;
        if (mLinear) {
            toCopyPtr = expandedAddr;
            indices[1] = b.getInt32(LinearBaseAddress);
            Value * baseAddrField = b.CreateInBoundsGEP(handleTy, handle, indices);
            Value * const baseAddr = b.CreateAlignedLoad(i8PtrTy, baseAddrField, voidPtrTyAlign);
            unreadDataPtr = b.CreateInBoundsGEP(b.getInt8Ty(), baseAddr, b.CreateMul(consumedChunks, bytesPerChunk));
        } else {
            Value * const newBufferOffset = b.CreateURem(consumedChunks, expandedCapacity);
            toCopyPtr = b.CreateInBoundsGEP(b.getInt8Ty(), expandedAddr, b.CreateMul(newBufferOffset, bytesPerChunk));
            Value * const oldBufferOffset = b.CreateURem(consumedChunks, initialCapacity);
            unreadDataPtr = b.CreateInBoundsGEP(b.getInt8Ty(), initialAddr, b.CreateMul(oldBufferOffset, bytesPerChunk));
        }

        if (LLVM_LIKELY(b.supportsIndirectBr())) {

            FixedArray<Constant *, DUFF_STEPS> copyLoopAddr;
            for (unsigned i = 0; i < DUFF_STEPS; ++i) {
                copyLoopAddr[i] = BlockAddress::get(f, copyLoopEntryPoint[(DUFF_STEPS - 1U) - i]);
            }

            ArrayType * const copyLoopArrayAddrTy = ArrayType::get(i8PtrTy, DUFF_STEPS);
            Constant * const copyLoopAddrArray = ConstantArray::get(copyLoopArrayAddrTy, copyLoopAddr);

            GlobalVariable * const copyLoopTargetArray =
                new GlobalVariable(*m, copyLoopArrayAddrTy, true, GlobalValue::ExternalLinkage, copyLoopAddrArray);

            FixedVectorType * const vecTy = b.getBitBlockType();
            PointerType * const vecPtrTy = vecTy->getPointerTo(mAddressSpace);


            const auto vecAlign = DL.getABITypeAlign(vecTy).value();

            const auto i8PtrAlign = DL.getABITypeAlign(i8PtrTy).value();

            Value * const maxNumOfBlocks = b.CreateMul(unreadChunks, blocksPerChunk);

            FixedArray<Value *, 2> jumpIndex;
            jumpIndex[0] = sz_ZERO;
            jumpIndex[1] = b.CreateAnd(maxNumOfBlocks, b.getSize(DUFF_STEPS - 1U));

            Value * const baseInput = b.CreatePointerCast(unreadDataPtr, vecPtrTy);
            Value * const baseOutput = b.CreatePointerCast(toCopyPtr, vecPtrTy);

            Value * const initialJumpTargetPtr = b.CreateGEP(copyLoopArrayAddrTy, copyLoopTargetArray, jumpIndex);
            Value * const initialJumpTarget = b.CreateAlignedLoad(i8PtrTy, initialJumpTargetPtr, i8PtrAlign);

            IndirectBrInst * const br = b.CreateIndirectBr(initialJumpTarget, DUFF_STEPS);

            FixedArray<PHINode *, DUFF_STEPS> copyAddrIndexPhi;
            for (unsigned i = 0; i < DUFF_STEPS; ++i) {
                copyAddrIndexPhi[i] = PHINode::Create(intPtrTy, 2, "", copyLoopEntryPoint[i]);
                copyAddrIndexPhi[i]->addIncoming(sz_ZERO, copyEntry);
                br->addDestination(copyLoopEntryPoint[i]);
            }

            for (unsigned i = 0; i < DUFF_STEPS; ++i) {
                b.SetInsertPoint(copyLoopEntryPoint[i]);
                Value * inputPtr = b.CreateGEP(vecTy, baseInput, copyAddrIndexPhi[i]);
                Value * outputPtr = b.CreateGEP(vecTy, baseOutput, copyAddrIndexPhi[i]);
                Value * in = b.CreateAlignedLoad(vecTy, inputPtr, vecAlign);
                b.CreateAlignedStore(in, outputPtr, vecAlign);
                Value * next = b.CreateAdd(copyAddrIndexPhi[i], sz_ONE);
                const auto k = (i + 1U) % DUFF_STEPS;
                copyAddrIndexPhi[k]->addIncoming(next, copyLoopEntryPoint[i]);
                if (k == 0) {
                    Value * const isDone = b.CreateICmpEQ(copyAddrIndexPhi[i], maxNumOfBlocks);
                    b.CreateCondBr(isDone, copyExit, copyLoopEntryPoint[0]);
                } else {
                    b.CreateBr(copyLoopEntryPoint[k]);
                }
            }

        } else {

            Value * const remainingBytes = b.CreateMul(unreadChunks, bytesPerChunk);
            b.CreateMemCpy(toCopyPtr, unreadDataPtr, remainingBytes, blockWidth / 8);

            b.CreateBr(copyExit);
        }

        b.SetInsertPoint(copyExit);
        if (mLinear) {
            b.CreateAlignedStore(b.CreatePointerCast(expandedAddr, addrPtrTy), addrField, voidPtrTyAlign);
            b.CreateAlignedStore(expandedCapacity, capacityField, intPtrTyAlign);
        } else {
            Value * const bSelector = b.CreateICmpEQ(selector, b.getSize(0));
            Value * nextAddrField = b.CreateSelect(bSelector, secondaryAddrField, primaryAddrField);
            Value * nextCapacityField = b.CreateSelect(bSelector, secondaryCapacityField, primaryCapacityField);
            b.CreateAlignedStore(b.CreatePointerCast(expandedAddr, addrPtrTy), nextAddrField, voidPtrTyAlign);
            b.CreateAlignedStore(expandedCapacity, nextCapacityField, intPtrTyAlign);
            Value * const nextSelector = b.CreateXor(selector, b.getSize(1));
            b.CreateAlignedStore(nextSelector, selectorField, intPtrTyAlign);
        }

        // now check if there is anything to delete
        indices[1] = b.getInt32(ThreadLocalField::PriorAddress);
        Value * const priorBufferField = b.CreateInBoundsGEP(threadLocalTy, threadLocalHandle, indices);
        Value * const priorBuffer = b.CreateAlignedLoad(i8PtrTy, priorBufferField, voidPtrTyAlign);

        indices[1] = b.getInt32(ThreadLocalField::PriorCapacity);
        Value * const priorCapacityField = b.CreateInBoundsGEP(threadLocalTy, threadLocalHandle, indices);
        Value * const priorCapacity = b.CreateAlignedLoad(intPtrTy, priorCapacityField, intPtrTyAlign);

        indices[1] = b.getInt32(ThreadLocalField::NewAddress);
        Value * const recentlyCreatedAddrField = b.CreateInBoundsGEP(threadLocalTy, threadLocalHandle, indices);
        Value * const recentlyCreatedAddr = b.CreateAlignedLoad(i8PtrTy, recentlyCreatedAddrField, voidPtrTyAlign);

        b.CallPrintInt("expandedAddr", expandedAddr);
        b.CallPrintInt("priorBuffer", priorBuffer);
        b.CallPrintInt("initialAddr", initialAddr);
        b.CallPrintInt("recentlyCreatedUnreleasedAddress", recentlyCreatedAddr);

        // If the user created a buffer in the same logical segment, free the old one as it hasn't been released
        // to the outer system. However, if we did just create it then the prior buffer we stored isn't safe to
        // delete yet.
        Value * wasRecentlyCreated = b.CreateICmpEQ(initialAddr, recentlyCreatedAddr);
        if (mLinear) {
            wasRecentlyCreated = b.CreateAnd(wasRecentlyCreated, b.CreateICmpNE(expandedBufferRetVal, sz_ZERO));
        }
        b.CallPrintInt("wasRecentlyCreated", wasRecentlyCreated);
        Value * const hasPendingPrior = b.CreateICmpNE(priorBuffer, nullAddrPtr);
        b.CreateUnlikelyCondBr(b.CreateOr(wasRecentlyCreated, hasPendingPrior), freeExistingBuffer, afterExpandInternalBuffer);

        b.SetInsertPoint(freeExistingBuffer);

        Value * const toDestroyAddr = b.CreateSelect(wasRecentlyCreated, initialAddr, priorBuffer);
        Value * const toDestroyCapacity = b.CreateSelect(wasRecentlyCreated, initialCapacity, priorCapacity);

        b.CallPrintInt("toDestroyAddr", toDestroyAddr);

        destroyBuffer(b, toDestroyAddr, toDestroyCapacity);

        b.CreateBr(afterExpandInternalBuffer);

        b.SetInsertPoint(afterExpandInternalBuffer);
        PHINode * const priorAllocAddrPhi = b.CreatePHI(i8PtrTy, 2);
        priorAllocAddrPhi->addIncoming(initialAddr, copyExit);
        priorAllocAddrPhi->addIncoming(nullAddrPtr, freeExistingBuffer);

        PHINode * const priorAllocCapacityPhi = b.CreatePHI(intPtrTy, 2);
        priorAllocCapacityPhi->addIncoming(initialCapacity, copyExit);
        priorAllocCapacityPhi->addIncoming(sz_ZERO, freeExistingBuffer);

        b.CreateAlignedStore(priorAllocAddrPhi, priorBufferField, voidPtrTyAlign);
        b.CreateAlignedStore(priorAllocCapacityPhi, priorCapacityField, intPtrTyAlign);
        b.CreateAlignedStore(expandedAddr, recentlyCreatedAddrField, voidPtrTyAlign);

        b.CreateBr(exit);

        b.SetInsertPoint(exit);
        if (mLinear) {
            Value * const consumedOffset = b.CreateSub(b.CreateURem(consumedChunks, expandedCapacity), consumedChunks);

            Value * const bytesPerChunk = b.CreateShl(blocksPerChunk, log2BytesPerBlock);

            Value * const newVirtualAddress = b.CreateInBoundsGEP(b.getInt8Ty(), expandedAddr, b.CreateMul(consumedOffset, bytesPerChunk));

            indices[1] = b.getInt32(LinearBaseAddress);
            Value * baseAddrField = b.CreateInBoundsGEP(handleTy, handle, indices);
            b.CreateAlignedStore(b.CreatePointerCast(newVirtualAddress, addrPtrTy), baseAddrField, voidPtrTyAlign);


            indices[1] = b.getInt32(LinearEffectiveCapacity);
            Value * baseCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
            b.CreateAlignedStore(requiredChunks, baseCapacityField, intPtrTyAlign);
        }
        assert (mLinear ^ (expandedBufferRetVal == nullptr));
        b.CreateRet(expandedBufferRetVal);
        b.restoreIP(ip);

    }

    FixedArray<Value *, 6> args;
    args[0] = b.CreatePointerCast(mHandle, voidPtrTy);
    args[1] = b.CreatePointerCast(mThreadLocalHandle, voidPtrTy);
    args[2] = produced;
    args[3] = consumed;
    args[4] = required;
    const auto typeSize = b.getTypeSize(DL, mType);
    assert (typeSize > 0);
    assert ((typeSize % (blockWidth / 8)) == 0);
    args[5] = b.getSize((8 * typeSize) / blockWidth);
    Value * const retVal = b.CreateCall(f, args);
    if (mLinear) {
        return retVal;
    } else {
        return sz_ONE;
    }
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

Value * RepeatingBuffer::reserveCapacity(KernelBuilder & b, Value * produced, Value * consumed, Value * const required) const  {
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
                                           llvm::Type * const type, const bool linear, const unsigned AddressSpace)
: InternalBuffer(id, BufferKind::ManagedDynamicBuffer, b, type, linear, AddressSpace) {

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
