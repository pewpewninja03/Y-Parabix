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

    assert ((size % getPageSize()) == 0);

    const auto memfd = create_memfd();
    if (memfd == -1) {
        std::string msg = "failed to create memfd, errno = " + std::to_string(errno);
        llvm::report_fatal_error(llvm::StringRef(msg));
    }

    if (ftruncate(memfd, size) == -1) {
        llvm::report_fatal_error(llvm::Twine{"failed to size mmap buffer to ", std::to_string(size)});
    }

    const size_t n = hasUnderflow ? 1 : 0;
    const size_t m = 2 + n;

    uint8_t * base = (uint8_t*)mmap(NULL, m * size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (base == MAP_FAILED) {
        llvm::report_fatal_error("failed to allocate virtual address space");
    }

    for (size_t i = 0; i < m; ++i) {
        void * p = mmap(base + (i * size), size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, memfd, 0);
        if (p == MAP_FAILED) {
            llvm::report_fatal_error("failed to map virtual address space");
        }
    }
    close(memfd);
    return base + (n * size);
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

namespace llvm { class Constant; }
namespace llvm { class Function; }

using namespace llvm;
using namespace IDISA;
using IDISA::IDISA_Builder;

using boost::intrusive::detail::is_pow2;
using boost::intrusive::detail::floor_log2;

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

void StreamSetBuffer::assertValidStreamIndex(kernel::KernelBuilder & b, Value * streamIndex) const {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value * const count = getStreamSetCount(b);
        Value * const index = b.CreateZExtOrTrunc(streamIndex, count->getType());
        Value * const withinSet = b.CreateICmpULT(index, count);
        b.CreateAssert(withinSet, "out-of-bounds stream access: %i of %i", index, count);
    }
}

Value * StreamSetBuffer::getStreamBlockPtr(kernel::KernelBuilder & b, Value * const baseAddress, Value * const streamIndex, Value * const blockIndex) const {
   // assertValidStreamIndex(b, streamIndex);
    return b.CreateInBoundsGEP(mType, baseAddress, {blockIndex, streamIndex});
}

Value * StreamSetBuffer::getStreamPackPtr(kernel::KernelBuilder & b, Value * const baseAddress, Value * const streamIndex, Value * blockIndex, Value * const packIndex) const {
   // assertValidStreamIndex(b, streamIndex);
    return b.CreateInBoundsGEP(mType, baseAddress, {blockIndex, streamIndex, packIndex});
}

llvm::Value * StreamSetBuffer::loadStreamBlock(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex, const bool unaligned) const {
    Value * addr = getStreamBlockPtr(b, baseAddress, streamIndex, blockIndex);
    return b.CreateAlignedLoad(b.getBitBlockType(), addr, unaligned ? 1U : (b.getBitBlockWidth() / 8));
}

llvm::Value * StreamSetBuffer::loadStreamPack(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex, llvm::Value * packIndex, const bool unaligned) const {
    Value * addr = getStreamPackPtr(b, baseAddress, streamIndex, blockIndex, packIndex);
    return b.CreateAlignedLoad(b.getBitBlockType(), addr, unaligned ? 1U : (b.getBitBlockWidth() / 8));
}

Value * StreamSetBuffer::getStreamSetCount(kernel::KernelBuilder & b) const {
    size_t count = 1;
    if (isa<ArrayType>(getBaseType())) {
        count = getBaseType()->getArrayNumElements();
    }
    return b.getSize(count);
}

bool StreamSetBuffer::isEmptySet() const {
    return getArraySize(mBaseType) == 0;
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
Value * StreamSetBuffer::getRawItemPointer(kernel::KernelBuilder & b, Value * streamIndex, Value * absolutePosition) const {
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
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            b.CreateAssertZero(b.CreateURemRational(absolutePosition, itemsPerByte),
                                "absolutePosition (%" PRIu64 " * %" PRIu64 "x%" PRIu64 ") must be byte aligned",
                                absolutePosition, streamCount, b.getSize(itemWidth));
        }
        pos = b.CreateUDivRational(pos, itemsPerByte);
        itemTy = b.getInt8Ty();
    }
    addr = b.CreatePointerCast(addr, itemTy->getPointerTo(mAddressSpace));
    return b.CreateInBoundsGEP(itemTy, addr, pos);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief resolveType
 ** ------------------------------------------------------------------------------------------------------------- */
Type * StreamSetBuffer::resolveType(kernel::KernelBuilder & b, Type * const streamSetType) {
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
 * @brief linkFunctions
 ** ------------------------------------------------------------------------------------------------------------- */
constexpr const char __MAKE_CIRCULAR_BUFFER[] =    "__make_circular_buffer";
constexpr const char __DESTROY_CIRCULAR_BUFFER[] = "__destroy_circular_buffer";

void StreamSetBuffer::linkFunctions(kernel::KernelBuilder & b) {
    b.LinkFunction(StringRef{__MAKE_CIRCULAR_BUFFER, strlen(__MAKE_CIRCULAR_BUFFER)}, make_circular_buffer);
    b.LinkFunction(StringRef{__DESTROY_CIRCULAR_BUFFER, strlen(__DESTROY_CIRCULAR_BUFFER)}, destroy_circular_buffer);
}

// External Buffer

StructType * ExternalBuffer::getHandleType(kernel::KernelBuilder & b) const {
    if (mHandleType == nullptr) {
        FixedArray<Type *, 2> fields;
        fields[0] = getPointerType();
        fields[1] = b.getSizeTy();
        mHandleType = StructType::get(b.getContext(), fields);
    }
    return mHandleType;
}

void ExternalBuffer::allocateBuffer(kernel::KernelBuilder & /* b */, Value * const /* capacityMultiplier */) {
    unsupported("allocateBuffer", "External");
}

void ExternalBuffer::releaseBuffer(kernel::KernelBuilder & /* b */) const {
    // this buffer is not responsible for free-ing th data associated with it
}

void ExternalBuffer::destroyBuffer(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value *capacity) const {

}

void ExternalBuffer::setBaseAddress(kernel::KernelBuilder & b, Value * const addr) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    Value * const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(BaseAddress)});
    b.CreateStore(b.CreatePointerBitCastOrAddrSpaceCast(addr, getPointerType()), p);
}

Value * ExternalBuffer::getBaseAddress(kernel::KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling getBaseAddress");
    Value * const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(BaseAddress)});
    return b.CreateAlignedLoad(getPointerType(), p, sizeof(void*));
}

void ExternalBuffer::setCapacity(kernel::KernelBuilder & b, Value * const capacity) const {
    assert (mHandle && "has not been set prior to calling setCapacity");
    Value *  const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(EffectiveCapacity)});
    b.CreateAlignedStore(b.CreateZExt(capacity, b.getSizeTy()), p, sizeof(size_t));
}

Value * ExternalBuffer::getCapacity(kernel::KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling getCapacity");
    Value * const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(EffectiveCapacity)});
    return b.CreateLoad(b.getSizeTy(), p);
}

Value * ExternalBuffer::getInternalCapacity(kernel::KernelBuilder & b) const {
    return getCapacity(b);
}

Value * ExternalBuffer::modByCapacity(kernel::KernelBuilder & /* b */, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    return offset;
}

Value * ExternalBuffer::getLinearlyAccessibleItems(kernel::KernelBuilder & b, Value * const fromPosition, Value * const totalItems) const {
    return b.CreateSub(totalItems, fromPosition);
}

Value * ExternalBuffer::getLinearlyWritableItems(kernel::KernelBuilder & b, Value * const fromPosition, Value * const /* consumed */) const {
    assert (fromPosition);
    Value * const capacity = getCapacity(b);
    assert (fromPosition->getType() == capacity->getType());
    return b.CreateSub(capacity, fromPosition);
}

Value * ExternalBuffer::getVirtualBasePtr(kernel::KernelBuilder & b, Value * baseAddress, Value * const /* transferredItems */) const {
    Constant * const sz_ZERO = b.getSize(0);
    Value * const addr = StreamSetBuffer::getStreamBlockPtr(b, baseAddress, sz_ZERO, sz_ZERO);
    return b.CreatePointerCast(addr, getPointerType());
}

inline void ExternalBuffer::assertValidBlockIndex(kernel::KernelBuilder & b, Value * blockIndex) const {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value * const blockCount = b.CreateCeilUDiv(getCapacity(b), b.getSize(b.getBitBlockWidth()));
        blockIndex = b.CreateZExtOrTrunc(blockIndex, blockCount->getType());
        Value * const withinCapacity = b.CreateICmpULT(blockIndex, blockCount);
        b.CreateAssert(withinCapacity, "blockIndex exceeds buffer capacity");
    }
}

Value * ExternalBuffer::requiresExpansion(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    unsupported("requiresExpansion", "External");
}

void ExternalBuffer::linearCopyBack(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    unsupported("linearCopyBack", "External");
}

Value * ExternalBuffer::expandBuffer(kernel::KernelBuilder & /* b */, Value * /* produced */, Value * /* consumed */, Value * const /* required */) const  {
    unsupported("expandBuffer", "External");
}

Value * ExternalBuffer::getMallocAddress(kernel::KernelBuilder & /* b */) const {
    unsupported("getMallocAddress", "External");
}

// Internal Buffer

Value * InternalBuffer::getStreamBlockPtr(kernel::KernelBuilder & b, Value * const baseAddress, Value * const streamIndex, Value * const blockIndex) const {
    Value * offset = nullptr;
    if (mLinear) {
        offset = blockIndex;
    } else {
        offset = modByCapacity(b, blockIndex);
    }
    return StreamSetBuffer::getStreamBlockPtr(b, baseAddress, streamIndex, offset);
}

Value * InternalBuffer::getStreamPackPtr(kernel::KernelBuilder & b, Value * const baseAddress, Value * const streamIndex, Value * const blockIndex, Value * const packIndex) const {
    Value * offset = nullptr;
    if (mLinear) {
        offset = blockIndex;
    } else {
        offset = modByCapacity(b, blockIndex);
    }
    return StreamSetBuffer::getStreamPackPtr(b, baseAddress, streamIndex, offset, packIndex);
}

Value * InternalBuffer::getVirtualBasePtr(kernel::KernelBuilder & b, Value * const baseAddress, Value * const transferredItems) const {
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

Value * InternalBuffer::getLinearlyAccessibleItems(kernel::KernelBuilder & b, Value * const processedItems, Value * const totalItems) const {
    return b.CreateSub(totalItems, processedItems);
}

Value * InternalBuffer::getLinearlyWritableItems(kernel::KernelBuilder & b, Value * const producedItems, Value * const consumedItems) const {
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

// Static Buffer

StructType * StaticBuffer::getHandleType(kernel::KernelBuilder & b) const {
    if (mHandleType == nullptr) {
        auto & C = b.getContext();
        PointerType * const typePtr = getPointerType();
        FixedArray<Type *, 4> types;
        types[BaseAddress] = typePtr;
        IntegerType * const sizeTy = b.getSizeTy();
        if (mLinear) {
            types[EffectiveCapacity] = sizeTy;
            types[MallocedAddress] = typePtr;
        } else {
            Type * const emptyTy = StructType::get(C);
            types[EffectiveCapacity] = emptyTy;
            types[MallocedAddress] = emptyTy;
        }
        types[InternalCapacity] = sizeTy;
        mHandleType = StructType::get(C, types);
    }
    return mHandleType;
}

void StaticBuffer::allocateBuffer(kernel::KernelBuilder & b, Value * const capacityMultiplier) {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * const handle = getHandle();
    assert (handle && "has not been set prior to calling allocateBuffer");
    Value * const capacity = b.CreateMul(capacityMultiplier, b.getSize(mCapacity));

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b.CreateAssert(capacity, "Static buffer capacity cannot be 0.");
    }

    indices[1] = b.getInt32(InternalCapacity);
    StructType * const handleTy = mHandleType;
    Value * const intCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
    b.CreateStore(capacity, intCapacityField);

    indices[1] = b.getInt32(BaseAddress);
    Value * const mallocAddr = b.CreatePageAlignedMalloc(mType, capacity, mAddressSpace);
    Value * const baseAddressField = b.CreateInBoundsGEP(handleTy, handle, indices);
    b.CreateStore(mallocAddr, baseAddressField);

    if (mLinear) {
        indices[1] = b.getInt32(EffectiveCapacity);
        Value * const capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
        b.CreateStore(capacity, capacityField);

        indices[1] = b.getInt32(MallocedAddress);
        Value * const concreteAddrField = b.CreateInBoundsGEP(handleTy, handle, indices);
        b.CreateStore(mallocAddr, concreteAddrField);
    }
}

void StaticBuffer::releaseBuffer(kernel::KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(mLinear ? MallocedAddress : BaseAddress);
    Value * const addressField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    Value * buffer = b.CreateLoad(mType, addressField);
    destroyBuffer(b, buffer, nullptr);
    b.CreateStore(ConstantPointerNull::get(cast<PointerType>(mType)), addressField);
}

void StaticBuffer::destroyBuffer(kernel::KernelBuilder & b, Value * baseAddress, llvm::Value *capacity) const {
    b.CreateFree(baseAddress);
}

Value * StaticBuffer::modByCapacity(kernel::KernelBuilder & b, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    if (LLVM_UNLIKELY(mLinear || isCapacityGuaranteed(offset, mCapacity))) {
        return offset;
    } else {
        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(0);
        indices[1] = b.getInt32(InternalCapacity);
        Value * ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const capacity = b.CreateLoad(b.getSizeTy(), ptr);
        return b.CreateURem(offset, capacity);
    }
}

Value * StaticBuffer::getCapacity(kernel::KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(mLinear ? EffectiveCapacity : InternalCapacity);
    Value * ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    Value * const capacity = b.CreateLoad(b.getSizeTy(), ptr);
    assert (capacity->getType()->isIntegerTy());
    return b.CreateMul(capacity, BLOCK_WIDTH, "capacity");
}

Value * StaticBuffer::getInternalCapacity(kernel::KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(InternalCapacity);
    Value * const intCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    Value * const capacity = b.CreateLoad(b.getSizeTy(), intCapacityField);
    assert (capacity->getType()->isIntegerTy());
    return b.CreateMul(capacity, BLOCK_WIDTH, "internalCapacity");
}

void StaticBuffer::setCapacity(kernel::KernelBuilder & b, Value * capacity) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(InternalCapacity);
    Value * const handle = getHandle(); assert (handle);
    Value * capacityField = b.CreateInBoundsGEP(mHandleType, handle, indices);
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    assert (capacity->getType()->isIntegerTy());
    Value * const cap = b.CreateExactUDiv(capacity, BLOCK_WIDTH);
    b.CreateStore(cap, capacityField);
    if (mLinear) {
        indices[1] = b.getInt32(EffectiveCapacity);
        Value * const effCapacityField = b.CreateInBoundsGEP(mHandleType, handle, indices);
        b.CreateStore(cap, effCapacityField);
    }
}

Value * StaticBuffer::getBaseAddress(kernel::KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    Value * const base = b.CreateInBoundsGEP(mHandleType, handle, indices);
    return b.CreateLoad(getPointerType(), base, "baseAddress");
}

void StaticBuffer::setBaseAddress(kernel::KernelBuilder & b, Value * addr) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    b.CreateStore(addr, b.CreateInBoundsGEP(mHandleType, handle, indices));
    if (mLinear) {
         indices[1] = b.getInt32(MallocedAddress);
         b.CreateStore(addr, b.CreateInBoundsGEP(mHandleType, handle, indices));
    }
}

Value * StaticBuffer::getMallocAddress(kernel::KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(mLinear ? MallocedAddress : BaseAddress);
    return b.CreateLoad(getPointerType(), b.CreateInBoundsGEP(mHandleType, getHandle(), indices));
}

Value * StaticBuffer::requiresExpansion(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    return b.getFalse();
}

void StaticBuffer::linearCopyBack(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    if (mLinear) {
        const auto blockWidth = b.getBitBlockWidth();
        assert (is_pow2(blockWidth));

        ConstantInt * const BLOCK_WIDTH = b.getSize(blockWidth);

        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(0);
        indices[1] = b.getInt32(EffectiveCapacity);
        Value * const capacityField = b.CreateInBoundsGEP(mHandleType, mHandle, indices);
        Value * consumedChunks = b.CreateUDiv(consumed, BLOCK_WIDTH);

        indices[1] = b.getInt32(MallocedAddress);
        Value * const mallocedAddrField = b.CreateInBoundsGEP(mHandleType, mHandle, indices);
        Value * const bufferStart = b.CreateLoad(getPointerType(), mallocedAddrField);
        assert (bufferStart->getType()->isPointerTy());
        Value * const newBaseAddress = b.CreateGEP(getPointerType(), bufferStart, b.CreateNeg(consumedChunks));
        Value * const effectiveCapacity = b.CreateAdd(consumedChunks, b.getSize(mCapacity));

        indices[1] = b.getInt32(BaseAddress);
        Value * const baseAddrField = b.CreateInBoundsGEP(mHandleType, mHandle, indices);

        b.CreateStore(newBaseAddress, baseAddrField);
        b.CreateStore(effectiveCapacity, capacityField);
    }
}

Value * StaticBuffer::expandBuffer(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * const required) const  {
    if (mLinear) {

        SmallVector<char, 200> buf;
        raw_svector_ostream name(buf);

        assert ("unspecified module" && b.getModule());

        name << "__StaticLinearBuffer_linearCopyBack_";

        Type * ty = getBaseType();
        const auto streamCount = ty->getArrayNumElements();
        name << streamCount << 'x';
        ty = ty->getArrayElementType();
        ty = cast<FixedVectorType>(ty)->getElementType();;
        const auto itemWidth = ty->getIntegerBitWidth();
        name << itemWidth << '_' << mAddressSpace;

        Value * const myHandle = getHandle();


        Module * const m = b.getModule();
        IntegerType * const sizeTy = b.getSizeTy();
        FunctionType * funcTy = FunctionType::get(b.getVoidTy(), {myHandle->getType(), sizeTy, sizeTy}, false);
        Function * func = m->getFunction(name.str());
        if (func == nullptr) {

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

            ConstantInt * const BLOCK_WIDTH = b.getSize(blockWidth);
            Constant * const CHUNK_SIZE = b.getTypeSize(mType);

            FixedArray<Value *, 2> indices;
            indices[0] = b.getInt32(0);

            Value * const consumedChunks = b.CreateUDiv(consumed, BLOCK_WIDTH);
            Value * const producedChunks = b.CreateCeilUDiv(produced, BLOCK_WIDTH);
            Value * const unconsumedChunks = b.CreateSub(producedChunks, consumedChunks);

            StructType * const handleTy = getHandleType(b);
            IntegerType * const sizeTy = b.getSizeTy();

            indices[1] = b.getInt32(BaseAddress);
            Value * const virtualBaseField = b.CreateInBoundsGEP(handleTy, handle, indices);
            Value * const virtualBase = b.CreateLoad(getPointerType(), virtualBaseField);

            indices[1] = b.getInt32(MallocedAddress);
            Value * const mallocAddrField = b.CreateInBoundsGEP(handleTy, handle, indices);
            Value * const mallocAddress = b.CreateLoad(getPointerType(), mallocAddrField);
            Value * const bytesToCopy = b.CreateMul(unconsumedChunks, CHUNK_SIZE);
            Value * const unreadDataPtr = b.CreateInBoundsGEP(mType, virtualBase, consumedChunks);

            indices[1] = b.getInt32(InternalCapacity);
            Value * const intCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
            Value * const bufferCapacity = b.CreateLoad(sizeTy, intCapacityField);

            b.CreateMemCpy(mallocAddress, unreadDataPtr, bytesToCopy, blockSize);

            Value * const newBaseAddress = b.CreateGEP(mType, mallocAddress, b.CreateNeg(consumedChunks));
            b.CreateStore(newBaseAddress, virtualBaseField);

            indices[1] = b.getInt32(EffectiveCapacity);

            Value * const capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
            Value * const effectiveCapacity = b.CreateAdd(consumedChunks, bufferCapacity);
            b.CreateStore(effectiveCapacity, capacityField);
            b.CreateRetVoid();

            b.restoreIP(ip);
            setHandle(myHandle);
        }

        b.CreateCall(funcTy, func, { myHandle, produced, consumed });
    }

    return nullptr;
}

// Dynamic Buffer

StructType * DynamicBuffer::getHandleType(kernel::KernelBuilder & b) const {
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

void DynamicBuffer::allocateBuffer(kernel::KernelBuilder & b, Value * const capacityMultiplier) {
    assert (mHandle && "has not been set prior to calling allocateBuffer");
    // note: when adding extensible stream sets, make sure to set the initial count here.
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);

    Value * const handle = getHandle();
    Value * capacity = b.CreateMul(capacityMultiplier, b.getSize(mInitialCapacity));

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b.CreateAssert(capacity, "Dynamic buffer capacity cannot be 0.");
    }


    if (mLinear) {

        Value * baseAddress = b.CreatePageAlignedMalloc(mType, capacity, mAddressSpace);

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
        b.CreateStore(capacity, capacityField);
        b.CreateStore(baseAddress, initialField);
        b.CreateStore(capacity, effCapacityField);

    } else {

        Module * m = b.getModule();
        DataLayout DL(m);

        const auto typeSize = b.getTypeSize(DL, mType);
        Rational stridesPerPage{getPageSize(), typeSize};
        capacity = b.CreateRoundUpRational(capacity, stridesPerPage.numerator());
        Value * capacityBytes = b.CreateMul(b.getSize(typeSize), capacity);

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
        b.CreateStore(capacity, capacityField);

        b.CreateStore(baseAddress, secAddressField);
        b.CreateStore(capacity, secCapacityField);

    }

}

void DynamicBuffer::releaseBuffer(kernel::KernelBuilder & b) const {
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

void DynamicBuffer::destroyBuffer(kernel::KernelBuilder & b, Value * baseAddress, Value * capacity) const {
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

void DynamicBuffer::setBaseAddress(kernel::KernelBuilder & /* b */, Value * /* addr */) const {
    unsupported("setBaseAddress", "Dynamic");
}

Value * DynamicBuffer::getBaseAddress(kernel::KernelBuilder & b) const {
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

Value * DynamicBuffer::getMallocAddress(kernel::KernelBuilder & b) const {
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

Value * DynamicBuffer::modByCapacity(kernel::KernelBuilder & b, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    if (mLinear || isCapacityGuaranteed(offset, mInitialCapacity)) {
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

Value * DynamicBuffer::getCapacity(kernel::KernelBuilder & b) const {
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

Value * DynamicBuffer::getInternalCapacity(kernel::KernelBuilder & b) const {
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

void DynamicBuffer::setCapacity(kernel::KernelBuilder & /* b */, Value * /* capacity */) const {
    unsupported("setCapacity", "Dynamic");
}

Value * DynamicBuffer::requiresExpansion(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {

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

void DynamicBuffer::linearCopyBack(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {

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

Value * DynamicBuffer::expandBuffer(kernel::KernelBuilder & b, Value * const produced, Value * const consumed, Value * const required) const {

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
            makeArgs[1] = b.getSize(mHasUnderflow);

            Value * newBuffer = b.CreateCall(m->getFunction(__MAKE_CIRCULAR_BUFFER), makeArgs);
            newBuffer = b.CreatePointerCast(newBuffer, typePtrTy);

            Value * newBufferOffset = b.CreateURem(consumedChunks, newInternalCapacity);
            Value * const toWriteDataPtr = b.CreateInBoundsGEP(mType, newBuffer, newBufferOffset);


            Value * const oldBufferOffset = b.CreateURem(consumedChunks, internalCapacity);
            Value * const unreadDataPtr = b.CreateInBoundsGEP(mType, virtualBase, oldBufferOffset);

            DataLayout DL(m);
            const auto typeSize = b.getTypeSize(DL, mType);
            const auto align = largestPowerOf2ThatDivides(typeSize);
            assert ((typeSize % align) == 0);

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

// MMapped Buffer

StructType * MMapedBuffer::getHandleType(kernel::KernelBuilder & b) const {
    if (mHandleType == nullptr) {
        auto & C = b.getContext();
        PointerType * const typePtr = getPointerType();
        IntegerType * const sizeTy = b.getSizeTy();
        FixedArray<Type *, 4> types;
        types[BaseAddress] = typePtr;
        types[Capacity] = sizeTy;
        types[Released] = sizeTy;
        types[Fd] = b.getInt32Ty();
        mHandleType = StructType::get(C, types);
    }
    return mHandleType;
}

void MMapedBuffer::allocateBuffer(kernel::KernelBuilder & b, Value * const capacityMultiplier) {
    assert (mHandle && "has not been set prior to calling allocateBuffer");
    // note: when adding extensible stream sets, make sure to set the initial count here.
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);

    Value * const handle = getHandle();
    Value * capacity = b.CreateMul(capacityMultiplier, b.getSize(mInitialCapacity));

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b.CreateAssert(capacity, "Dynamic buffer capacity cannot be 0.");
    }

    indices[1] = b.getInt32(BaseAddress);
    Value * const baseAddressField = b.CreateInBoundsGEP(mHandleType, handle, indices);

    Constant * const typeSize = b.getTypeSize(mType);
    Value * const minCapacity = b.CreateCeilUDiv(b.getSize(ANON_MMAP_SIZE), typeSize);

    capacity = b.CreateUMax(capacity, minCapacity);

    Value * const fileSize = b.CreateMul(typeSize, capacity);

    Value * const fd = b.CreateMemFdCreate(b.GetString("streamset"), b.getInt32(0));

    b.CreateFTruncate(fd, fileSize);

    PointerType * const voidPtrTy = b.getVoidPtrTy();
    ConstantInt * const prot =  b.getInt32(PROT_READ | PROT_WRITE);
    ConstantInt * const intflags =  b.getInt32(MAP_PRIVATE | MAP_NORESERVE);
    Constant * const sz_ZERO = b.getSize(0);
    Value * baseAddress = b.CreateMMap(ConstantPointerNull::getNullValue(voidPtrTy), fileSize, prot, intflags, fd, sz_ZERO);

    baseAddress = b.CreatePointerCast(baseAddress, mType->getPointerTo(mAddressSpace));

    b.CreateStore(baseAddress, baseAddressField);

    indices[1] = b.getInt32(Capacity);
    Value * const capacityField = b.CreateInBoundsGEP(mHandleType, handle, indices);
    b.CreateStore(capacity, capacityField);

    indices[1] = b.getInt32(Fd);
    Value * const fdField = b.CreateInBoundsGEP(mHandleType, handle, indices);
    b.CreateStore(fd, fdField);

}

void MMapedBuffer::releaseBuffer(kernel::KernelBuilder & b) const {
    /* Free the dynamically allocated buffer(s). */
    Value * const handle = getHandle();
    Constant * const typeSize = b.getTypeSize(mType);
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(BaseAddress);
    Value * const baseAddressField = b.CreateInBoundsGEP(mHandleType, handle, indices);
    Value * const baseAddress = b.CreateLoad(getPointerType(), baseAddressField);
    indices[1] = b.getInt32(Capacity);
    Value * const capacityField = b.CreateInBoundsGEP(mHandleType, handle, indices);
    Value * const capacity = b.CreateLoad(b.getSizeTy(), capacityField);
    Value * const fileSize = b.CreateMul(typeSize, capacity);
    b.CreateMUnmap(baseAddress, fileSize);
    b.CreateStore(ConstantPointerNull::get(getPointerType()), baseAddressField);
    indices[1] = b.getInt32(Fd);
    Value * const fdField = b.CreateInBoundsGEP(mHandleType, handle, indices);
    b.CreateCloseCall(b.CreateLoad(b.getInt32Ty(), fdField));
}

void MMapedBuffer::destroyBuffer(kernel::KernelBuilder & b, Value * baseAddress, Value *capacity) const {

}

void MMapedBuffer::setBaseAddress(kernel::KernelBuilder & /* b */, Value * /* addr */) const {
    unsupported("setBaseAddress", "MMaped");
}

Value * MMapedBuffer::getBaseAddress(kernel::KernelBuilder & b) const {
    assert (getHandle());
    Value * const ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), {b.getInt32(0), b.getInt32(BaseAddress)});
    return b.CreateLoad(getPointerType(), ptr);
}

Value * MMapedBuffer::getMallocAddress(kernel::KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(BaseAddress);
    return b.CreateLoad(getPointerType(), b.CreateInBoundsGEP(mHandleType, getHandle(), indices));
}

Value * MMapedBuffer::modByCapacity(kernel::KernelBuilder & b, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    return offset;
}

Value * MMapedBuffer::getCapacity(kernel::KernelBuilder & b) const {
    assert (getHandle());
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(Capacity);
    Value * ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    Value * const capacity = b.CreateLoad(b.getSizeTy(), ptr);
    assert (capacity->getType()->isIntegerTy());
    return b.CreateMul(capacity, BLOCK_WIDTH, "capacity");
}

Value * MMapedBuffer::getInternalCapacity(kernel::KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(Capacity);
    Value * const intCapacityField = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    Value * const capacity = b.CreateLoad(b.getSizeTy(), intCapacityField);
    assert (capacity->getType()->isIntegerTy());
    return b.CreateMul(capacity, BLOCK_WIDTH);
}

void MMapedBuffer::setCapacity(kernel::KernelBuilder & /* b */, Value * /* capacity */) const {
    unsupported("setCapacity", "MMaped");
}

Value * MMapedBuffer::requiresExpansion(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {

    assert (mLinear);

    const auto blockWidth = b.getBitBlockWidth();
    assert (is_pow2(blockWidth));
    ConstantInt * const BLOCK_WIDTH = b.getSize(blockWidth);

    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(BaseAddress);
    Value * const virtualBaseField = b.CreateInBoundsGEP(mHandleType, mHandle, indices);
    Value * const virtualBase = b.CreateLoad(getPointerType(), virtualBaseField);
    Value * const consumedChunks = b.CreateUDiv(consumed, BLOCK_WIDTH);

    DataLayout DL(b.getModule());
    Type * const intPtrTy = DL.getIntPtrType(virtualBase->getType());
    Value * const virtualBaseInt = b.CreatePtrToInt(virtualBase, intPtrTy);
    Value * startOfUsedBuffer = b.CreatePtrToInt(b.CreateInBoundsGEP(mType, virtualBase, consumedChunks), intPtrTy);
    Value * unnecessaryBytes = b.CreateSub(startOfUsedBuffer, virtualBaseInt);
    unnecessaryBytes = b.CreateRoundDown(unnecessaryBytes, b.getSize(getPageSize()));
    // assume that we can always discard memory
    b.CreateMAdvise(virtualBase, unnecessaryBytes, MADV_DONTNEED);

    indices[1] = b.getInt32(Capacity);
    Value * const capacityField = b.CreateInBoundsGEP(mHandleType, mHandle, indices);
    Value * const capacity = b.CreateLoad(b.getSizeTy(), capacityField);

    return b.CreateICmpUGE(b.CreateAdd(produced, required), b.CreateMul(capacity, BLOCK_WIDTH));

}

void MMapedBuffer::linearCopyBack(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    /* do nothing */
}


Value * MMapedBuffer::expandBuffer(kernel::KernelBuilder & b, Value * const produced, Value * const consumed, Value * const required) const {

    Value * const handle = getHandle();

    Constant * const typeSize = b.getTypeSize(mType);
    Value * const expandStepSize = b.CreateCeilUDiv(b.getSize(ANON_MMAP_SIZE), typeSize);

    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(Capacity);
    Value * const capacityField = b.CreateInBoundsGEP(mHandleType, handle, indices);
    const auto blockWidth = b.getBitBlockWidth();
    assert (is_pow2(blockWidth));
    ConstantInt * const BLOCK_WIDTH = b.getSize(blockWidth);

    Value * newCapacity = b.CreateCeilUDiv(b.CreateAdd(produced, required), BLOCK_WIDTH);
    newCapacity = b.CreateRoundUp(newCapacity, expandStepSize);
    b.CreateStore(newCapacity, capacityField);

    indices[1] = b.getInt32(Fd);
    Value * const fdField = b.CreateInBoundsGEP(mHandleType, handle, indices);
    Value * const fd = b.CreateLoad(b.getInt32Ty(), fdField);
    b.CreateFTruncate(fd, b.CreateMul(newCapacity, typeSize));

    return nullptr;
}

// Repeating Buffer

StructType * RepeatingBuffer::getHandleType(kernel::KernelBuilder & b) const {
    if (mHandleType == nullptr) {
        auto & C = b.getContext();
        FixedArray<Type *, 1> types;
        types[BaseAddress] = getPointerType();
        mHandleType = StructType::get(C, types);
    }
    return mHandleType;
}

void RepeatingBuffer::allocateBuffer(kernel::KernelBuilder & b, Value * const capacityMultiplier) {
    unsupported("allocateBuffer", "Repeating");
}

void RepeatingBuffer::releaseBuffer(kernel::KernelBuilder & b) const {
    unsupported("releaseBuffer", "Repeating");
}

void RepeatingBuffer::destroyBuffer(kernel::KernelBuilder & b, Value * baseAddress, llvm::Value *capacity) const {

}

Value * RepeatingBuffer::modByCapacity(kernel::KernelBuilder & b, Value * const offset) const {
    Value * const capacity = b.CreateExactUDiv(mModulus, b.getSize(b.getBitBlockWidth()));
    return b.CreateURem(offset, capacity);
}

Value * RepeatingBuffer::getCapacity(kernel::KernelBuilder & b) const {
    return mModulus;
}

Value * RepeatingBuffer::getInternalCapacity(kernel::KernelBuilder & b) const {
    return mModulus;
}

void RepeatingBuffer::setCapacity(kernel::KernelBuilder & b, Value * capacity) const {
    unsupported("setCapacity", "Repeating");
}


Value * RepeatingBuffer::getVirtualBasePtr(kernel::KernelBuilder & b, Value * const baseAddress, Value * const transferredItems) const {
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

Value * RepeatingBuffer::getBaseAddress(kernel::KernelBuilder & b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    Value * const base = b.CreateInBoundsGEP(mHandleType, handle, indices);
    return b.CreateLoad(getPointerType(), base, "baseAddress");
}

void RepeatingBuffer::setBaseAddress(kernel::KernelBuilder & b, Value * addr) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    b.CreateStore(addr, b.CreateInBoundsGEP(mHandleType, handle, indices));
}

Value * RepeatingBuffer::getMallocAddress(kernel::KernelBuilder & b) const {
    return getBaseAddress(b);
}

Value * RepeatingBuffer::requiresExpansion(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    return b.getFalse();
}

void RepeatingBuffer::linearCopyBack(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * required) const {
    unsupported("linearCopyBack", "Repeating");
}

Value * RepeatingBuffer::expandBuffer(kernel::KernelBuilder & b, Value * produced, Value * consumed, Value * const required) const  {
    unsupported("linearCopyBack", "Repeating");
}

// Constructors

ExternalBuffer::ExternalBuffer(const unsigned id, kernel::KernelBuilder & b, Type * const type,
                               const bool linear,
                               const unsigned AddressSpace)
: StreamSetBuffer(id, BufferKind::ExternalBuffer, b, type,  linear, AddressSpace) {

}

StaticBuffer::StaticBuffer(const unsigned id, kernel::KernelBuilder & b, Type * const type,
                           const size_t capacity,
                           const bool linear, const unsigned AddressSpace)
: InternalBuffer(id, BufferKind::StaticBuffer, b, type, linear, AddressSpace)
, mCapacity(capacity) {
}

DynamicBuffer::DynamicBuffer(const unsigned id, kernel::KernelBuilder & b, Type * const type,
                             const size_t initialCapacity, const bool hasUnderflow,
                             const bool linear, const unsigned AddressSpace)
: InternalBuffer(id, BufferKind::DynamicBuffer, b, type, linear, AddressSpace)
, mInitialCapacity(initialCapacity)
, mHasUnderflow(hasUnderflow) {

}

MMapedBuffer::MMapedBuffer(const unsigned id, kernel::KernelBuilder & b, Type * const type,
                             const size_t initialCapacity, const size_t overflowSize, const size_t underflowSize,
                             const bool linear, const unsigned AddressSpace)
: InternalBuffer(id, BufferKind::MMapedBuffer, b, type, linear, AddressSpace)
, mInitialCapacity(initialCapacity) {

}

RepeatingBuffer::RepeatingBuffer(const unsigned id, kernel::KernelBuilder & b, Type * const type, const bool unaligned)
: InternalBuffer(id, BufferKind::RepeatingBuffer, b, type, false, 0)
, mUnaligned(unaligned) {

}


inline InternalBuffer::InternalBuffer(const unsigned id, const BufferKind k, kernel::KernelBuilder & b, Type * const baseType,
                                      const bool linear, const unsigned AddressSpace)
: StreamSetBuffer(id, k, b, baseType, linear, AddressSpace) {

}

inline StreamSetBuffer::StreamSetBuffer(const unsigned id, const BufferKind k, kernel::KernelBuilder & b, Type * const baseType,
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
