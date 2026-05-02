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

#define BEGIN_SCOPED_REGION {
#define END_SCOPED_REGION }

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

    uint8_t * const base = (uint8_t*)mmap(NULL, m * size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
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

uint8_t * make_fd_backed_buffer(const size_t size, int & memfd) {

    assert (size > 0);
    assert ((size % getPageSize()) == 0);

    memfd = create_memfd();
    if (memfd == -1) {
        SmallVector<char, 256> tmp;
        raw_svector_ostream msg(tmp);
        msg << "failed to create memfd (" << (size_t)errno << ')';
        report_fatal_error(msg.str());
    }

    if (ftruncate(memfd, size) != 0) {
        report_fatal_error(Twine{"failed to size mmap buffer to ", std::to_string(size)});
    }

    uint8_t * p = (uint8_t*)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
    if (p == MAP_FAILED) {
        report_fatal_error("failed to allocate virtual address space");
    }

    return p;
}

uint8_t * resize_fd_backed_buffer(const int memfd, uint8_t * const buffer, const size_t priorSize, const size_t newSize) {

    assert (newSize > 0);
    assert ((newSize % getPageSize()) == 0);

//    if (msync(buffer, priorSize, MS_SYNC) != 0) {
//        report_fatal_error(Twine{"failed to sync mmap buffer"});
//    }

    if (ftruncate(memfd, newSize) != 0) {
        report_fatal_error(Twine{"failed to resize mmap buffer to ", std::to_string(newSize)});
    }

    // TODO: mremap could potentially take the old buffer as a hint where to place the new one and reuse the addr space
    // but this function is not supplied on MacOS. Similarly, MAP_FIXED_NOREPLACE might work on Linux if defined?

    uint8_t * p = (uint8_t*)mmap(NULL, newSize, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
    if (p == MAP_FAILED) {
        report_fatal_error("failed to resize virtual address space");
    }

    return p;
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
            Value * cap = getInternalCapacity(b);
            Value * isZero = b.CreateICmpEQ(cap, ConstantInt::getNullValue(sizeTy));
            cap = b.CreateSelect(isZero, ConstantInt::getAllOnesValue(sizeTy), cap);
            pos = b.CreateURem(pos, cap);
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
 * @brief freePendingDeletions
 ** ------------------------------------------------------------------------------------------------------------- */
void StreamSetBuffer::freePendingDeletions(KernelBuilder & /* b */, llvm::Value * /* consumed */) const {

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isItemAlignedAccessWithinStreamSetMemory
 ** ------------------------------------------------------------------------------------------------------------- */
void StreamSetBuffer::assertAccessIsWithinStreamSetMemory(KernelBuilder & b, Constant * name, Value * ptr, const size_t size, Value * const start, Value * const end) const {

    assert (codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts));

    ConstantInt * sz_ZERO = b.getSize(0);
    ConstantInt * sz_LOG_2_BW = b.getSize(floor_log2(b.getBitBlockWidth()));

    Value * const ba = getBaseAddress(b);

    Value * startPtr = getStreamBlockPtr(b, ba, sz_ZERO, b.CreateLShr(start, sz_LOG_2_BW));
    Value * endPtr = getStreamBlockPtr(b, ba, sz_ZERO, b.CreateLShr(end, sz_LOG_2_BW));

    auto & dl = b.getModule()->getDataLayout();
    IntegerType * const intPtrTy = dl.getIntPtrType(b.getContext());
    startPtr = b.CreatePtrToInt(startPtr, intPtrTy);
    Value * ptrInt = b.CreatePtrToInt(ptr, intPtrTy);
    Value * valid = b.CreateICmpULE(startPtr, ptrInt);
    Value * outPtr = b.CreateAdd(ptrInt, ConstantInt::get(intPtrTy, size)); assert (size > 0);
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
constexpr const char __MAKE_FD_BACKED_BUFFER[] =    "__ss_mfb";
constexpr auto __MAKE_FD_BACKED_BUFFER_LENGTH = std::string_view(__MAKE_FD_BACKED_BUFFER).size();
constexpr const char __RESIZE_FD_BACKED_BUFFER[] =    "__ss_rfb";
constexpr auto ____RESIZE_FD_BACKED_BUFFER_LENGTH = std::string_view(__RESIZE_FD_BACKED_BUFFER).size();
constexpr const char __MUNMAP[] =    "munmap";
constexpr auto __MUNMAP_LENGTH = std::string_view(__MUNMAP).size();
constexpr const char __CLOSE[] =    "close";
constexpr auto __CLOSE_LENGTH = std::string_view(__CLOSE).size();

void StreamSetBuffer::linkFunctions(KernelBuilder & b) {
    IntegerType * const i32Ty = b.getInt32Ty();
    b.LinkFunction(StringRef{__MAKE_CIRCULAR_BUFFER, __MAKE_CIRCULAR_BUFFER_LENGTH}, make_circular_buffer);
    b.LinkFunction(StringRef{__MAKE_FD_BACKED_BUFFER, __MAKE_FD_BACKED_BUFFER_LENGTH}, make_fd_backed_buffer);
    b.LinkFunction(StringRef{__RESIZE_FD_BACKED_BUFFER, ____RESIZE_FD_BACKED_BUFFER_LENGTH}, resize_fd_backed_buffer);
    b.LinkFunction(StringRef{__MUNMAP, __MUNMAP_LENGTH}, munmap);
    BEGIN_SCOPED_REGION
    FixedArray<Type *, 1> params;
    params[0] = i32Ty;
    FunctionType * fCloseTy = FunctionType::get(i32Ty, params, false);
    b.LinkFunction(StringRef{__CLOSE, __CLOSE_LENGTH}, fCloseTy, (void*)close);
    END_SCOPED_REGION
}

// External Buffer

StructType * ExternalBuffer::getExternalHandleType(KernelBuilder & b) {
    FixedArray<Type *, 2> fields;
    fields[0] = b.getVoidPtrTy();
    fields[1] = b.getSizeTy();
    return StructType::get(b.getContext(), fields);
}

StructType * ExternalBuffer::getHandleType(KernelBuilder & b) const {
    if (mHandleType == nullptr) {
        mHandleType = getExternalHandleType(b);
    }
    return mHandleType;
}


void ExternalBuffer::allocateBuffer(KernelBuilder & /* b */, Value * const /* capacityMultiplier */, Value * reportCallback, Value * pipelineHandle, Value * portNum) {
    unsupported("allocateBuffer", "External");
}

void ExternalBuffer::releaseBuffer(KernelBuilder & /* b */) const {
    // this buffer is not responsible for free-ing th data associated with it
}

void ExternalBuffer::setBaseAddress(KernelBuilder & b, Value * const addr) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    Value * const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(BaseAddress)});
    b.CreateStore(b.CreatePointerBitCastOrAddrSpaceCast(addr, b.getVoidPtrTy()), p);
}

Value * ExternalBuffer::getBaseAddress(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling getBaseAddress");
    Value * const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(BaseAddress)});
    Module * const m = b.getModule();
    auto & DL = m->getDataLayout();
    PointerType * const voidPtrTy = b.getVoidPtrTy();
    const auto ptrTyAlign = DL.getABITypeAlign(voidPtrTy).value();
    return b.CreatePointerCast(b.CreateAlignedLoad(voidPtrTy, p, ptrTyAlign), getPointerType());
}

void ExternalBuffer::setCapacity(KernelBuilder & b, Value * const capacity) const {
    assert (mHandle && "has not been set prior to calling setCapacity");
    Value *  const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(EffectiveCapacity)});
    Module * const m = b.getModule();
    auto & DL = m->getDataLayout();
    IntegerType * sizeTy = b.getSizeTy();
    assert (capacity->getType() == sizeTy);
    const auto sizeTyAlign = DL.getABITypeAlign(sizeTy).value();
    b.CreateAlignedStore(capacity, p, sizeTyAlign);
}

Value * ExternalBuffer::getCapacity(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling getCapacity");
    Module * const m = b.getModule();
    auto & DL = m->getDataLayout();
    IntegerType * sizeTy = b.getSizeTy();
    const auto sizeTyAlign = DL.getABITypeAlign(sizeTy).value();
    Value * const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(EffectiveCapacity)});
    return b.CreateAlignedLoad(sizeTy, p, sizeTyAlign);
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

Value * ExternalBuffer::reserveCapacity(KernelBuilder & /* b */, Value * /* produced */, Value * /* consumed */, Value * const /* required */, Value * /* reportCallback */, Value * /* pipelineHandle */, Value * /* portNum */) const  {
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
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts, codegen::EnableStreamSetAsserts))) {
        Value * const valid = b.CreateICmpULE(consumedItems, producedItems);
        b.CreateAssert(valid, "consumed item count (%" PRIu64 ") exceeds produced (%" PRIu64 ").",
                        consumedItems, producedItems);
    }
    Value * const capacity = getInternalCapacity(b);
    Value * const unconsumedItems = b.CreateSub(producedItems, consumedItems);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts, codegen::EnableStreamSetAsserts))) {
        Value * const valid = b.CreateICmpULE(unconsumedItems, capacity);
        b.CreateAssert(valid, "unconsumed item count (%" PRIu64 ") exceeds capacity (%" PRIu64 ").",
                        unconsumedItems, capacity);
    }
    return b.CreateSub(capacity, unconsumedItems);
}

// Managed Dynamic Buffer

enum PendingDeletionField {
    PendingDeletionAddress = 0,
    PendingDeletionCapacity = 1,
    PendingDeletionConsumed = 2,
    PendingDeletionNextLink = 3
};

static StructType * makePendingDeletionStructTy(KernelBuilder & b) {
    auto & C = b.getContext();
    IntegerType * const sizeTy = b.getSizeTy();
    Type * const voidPtrTy = b.getVoidPtrTy();
    FixedArray<Type *, 3> fixedDeletionFields;
    fixedDeletionFields[PendingDeletionAddress] = voidPtrTy;
    fixedDeletionFields[PendingDeletionCapacity] = sizeTy;
    fixedDeletionFields[PendingDeletionConsumed] = sizeTy;
    StructType * const fixedDeletionTy = StructType::get(C, fixedDeletionFields);
    FixedArray<Type *, 2> pendingDeletionFields;
    pendingDeletionFields[0] = ArrayType::get(fixedDeletionTy, 2);
    pendingDeletionFields[1] = voidPtrTy; // PendingDeletionAdditionalStructPointer
    return StructType::get(C, pendingDeletionFields);
}


StructType * ManagedDynamicBuffer::getHandleType(KernelBuilder & b) const {
    if (mHandleType == nullptr) {
        auto & C = b.getContext();
        IntegerType * const sizeTy = b.getSizeTy();
        Type * const voidPtrTy = b.getVoidPtrTy();
        FixedArray<Type *, 4> fields;
        fields[MDB_Field::LinearMallocedAddress] = voidPtrTy;
        fields[MDB_Field::LinearInternalCapacity] = sizeTy;
        fields[MDB_Field::LinearBaseAddress] = voidPtrTy;
        fields[MDB_Field::PendingDeletionStruct] = makePendingDeletionStructTy(b);
        mHandleType = StructType::get(C, fields);
    }
    assert (&mHandleType->getContext() == &b.getContext());
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

void ManagedDynamicBuffer::allocateBuffer(KernelBuilder & b, Value * const capacityMultiplier, Value *reportCallback, Value *pipelineHandle, Value *portNum) {
    assert (mHandle && "has not been set prior to calling allocateBuffer");

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    assert ("unspecified module" && b.getModule());

    const auto traceDynamicBuffer = (reportCallback != nullptr);

    name << "__ManagedDynamicBuffer";
    if (mLinear) {
        name << "Linear";
    }
    name << "_initial_alloc" << mAddressSpace;
    if (LLVM_UNLIKELY(traceDynamicBuffer)) {
        name << 'T';
    }

    Module * const m = b.getModule();

    auto & DL = m->getDataLayout();
    Function * f = m->getFunction(name.str());
    Type * const voidPtrTy = b.getVoidPtrTy();

    if (f == nullptr) {

        LLVMContext & C = m->getContext();

        StructType * handleTy = getHandleType(b);
        PointerType * const handlePtrTy = handleTy->getPointerTo(mAddressSpace);


        PointerType * const addrPtrTy = b.getVoidPtrTy();
        IntegerType * const intPtrTy = b.getIntPtrTy(DL);

        const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();
        const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

        SmallVector<Type *, 6> paramTypes(traceDynamicBuffer ? 6 : 3);
        paramTypes[0] = voidPtrTy;
        paramTypes[1] = intPtrTy;
        paramTypes[2] = intPtrTy;
        if (LLVM_UNLIKELY(traceDynamicBuffer)) {
            paramTypes[3] = voidPtrTy;
            paramTypes[4] = voidPtrTy;
            paramTypes[5] = intPtrTy;
        }

        FunctionType * funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

        const auto ip = b.saveIP();

        f = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            f->setHasUWTable();
            #else
            f->setUWTableKind(UWTableKind::Default);
            #endif
        }

        BasicBlock * const entry = BasicBlock::Create(C, "entry", f);
        BasicBlock * const allocBuffer = BasicBlock::Create(C, "allocBuffer", f);
        BasicBlock * const exit = BasicBlock::Create(C, "exit", f);

        b.SetInsertPoint(entry);

        auto arg = f->arg_begin();
        auto nextArg = [&]() {
            assert (arg != f->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };

        Value * handle = nextArg();
        handle->setName("handle");
        handle = b.CreatePointerCast(handle, handlePtrTy);
        Value * capacity = nextArg();
        capacity->setName("capacity");
        Value * typeSize = nextArg();
        typeSize->setName("typeSize");
        Value * reportCallback = nullptr;
        Value * pipelineHandle = nullptr;
        Value * portNum = nullptr;
        if (LLVM_UNLIKELY(traceDynamicBuffer)) {
            reportCallback = nextArg();
            reportCallback->setName("reportCallback");
            pipelineHandle = nextArg();
            pipelineHandle->setName("pipelineHandle");
            portNum = nextArg();
            portNum->setName("portNum");
        }
        assert (arg == f->arg_end());

        ConstantInt * const i32_ZERO = b.getInt32(0);
        ConstantInt * const i32_ONE = b.getInt32(1);
        Constant * const nullVoidPtr = ConstantPointerNull::get(addrPtrTy);

        FixedArray<Value *, 2> indices;
        indices[0] = i32_ZERO;
        indices[1] = b.getInt32(LinearBaseAddress);

        Value * const baseAddressField = b.CreateInBoundsGEP(handleTy, handle, indices);
        Value * currentAddr = b.CreateAlignedLoad(addrPtrTy, baseAddressField, voidPtrTyAlign);

        // If the user has filled in a base address in the init function, assume they're handling all
        // memory management.
        b.CreateCondBr(b.CreateICmpEQ(currentAddr, nullVoidPtr), allocBuffer, exit);

        // --------------------------------------------------------

        b.SetInsertPoint(allocBuffer);
        Value * capacityBytes = b.CreateMul(typeSize, capacity);

        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            b.CreateAssert(capacity, "capacity cannot be 0.");
            b.CreateAssertZero(b.CreateURemRational(capacityBytes, getPageSize()), "%" PRIu64 " x %" PRIu64, typeSize, capacity);
        }

        ConstantInt * const sz_ZERO = b.getSize(0);

        Value * baseAddress = nullptr;

        if (mLinear) {
            baseAddress = b.CreatePageAlignedMalloc(capacityBytes);
        } else {
            FixedArray<Value *, 2> makeArgs;
            makeArgs[0] = capacityBytes;
            makeArgs[1] = sz_ZERO;
            Function * makeBuffer = m->getFunction(__MAKE_CIRCULAR_BUFFER); assert (makeBuffer);
            baseAddress = b.CreateCall(makeBuffer, makeArgs);
        }

        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * mallocAddrField = b.CreateInBoundsGEP(handleTy, handle, indices);
        b.CreateAlignedStore(baseAddress, mallocAddrField, voidPtrTyAlign);



        indices[1] = b.getInt32(LinearInternalCapacity);
        Value * capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
        b.CreateAlignedStore(capacity, capacityField, intPtrTyAlign);

        b.CreateAlignedStore(baseAddress, baseAddressField, voidPtrTyAlign);
        b.CreateBr(exit);

        // --------------------------------------------------------

        b.SetInsertPoint(exit);

        FixedArray<Value *, 5> indices5;
        indices5[0] = i32_ZERO;
        indices5[1] = b.getInt32(PendingDeletionStruct);
        indices5[2] = i32_ZERO;
        indices5[3] = i32_ZERO;
        indices5[4] = b.getInt32(PendingDeletionConsumed);

        Constant * const sz_ALL_ONES = ConstantInt::getAllOnesValue(intPtrTy);

        Value * const consumed0Field = b.CreateGEP(handleTy, handle, indices5);
        b.CreateAlignedStore(sz_ALL_ONES, consumed0Field, intPtrTyAlign);

        indices5[3] = i32_ONE;

        Value * const consumed1Field = b.CreateGEP(handleTy, handle, indices5);
        b.CreateAlignedStore(sz_ALL_ONES, consumed1Field, intPtrTyAlign);

        if (LLVM_UNLIKELY(traceDynamicBuffer)) {

            FixedArray<Type *, 4> paramTypes;
            paramTypes[0] = voidPtrTy; // pipeline handle
            paramTypes[1] = intPtrTy; // port num
            paramTypes[2] = intPtrTy; // produced
            paramTypes[3] = intPtrTy; // capacity

            FunctionType * funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

            FixedArray<Value *, 4> callbackArgs;
            callbackArgs[0] = pipelineHandle;
            callbackArgs[1] = portNum;
            callbackArgs[2] = sz_ZERO;
            callbackArgs[3] = b.CreateMul(capacity, b.getSize(b.getBitBlockWidth()));

            b.CreateCall(funcTy, b.CreatePointerCast(reportCallback, funcTy->getPointerTo()), callbackArgs);
        }

        b.CreateRetVoid();

        b.restoreIP(ip);
    }

    const auto typeSize = b.getTypeSize(DL, mType);
    assert (typeSize > 0);
    assert ((typeSize % (b.getBitBlockWidth() / 8)) == 0);

    // Let C be the capacity multiplyer, T be the type size and P be the page size.
    // We want to solve (C * T) mod P = 0, where T and P are constants but C isn't
    // known until after JIT-compilation.

    Rational stridesPerPage{getPageSize(), typeSize};
    Value * capacity = b.CreateRoundUpRational(capacityMultiplier, stridesPerPage.numerator());
    SmallVector<Value *, 6> args(traceDynamicBuffer ? 6 : 3);
    args[0] = b.CreatePointerCast(getHandle(), voidPtrTy);
    args[1] = capacity;
    args[2] = b.getSize(typeSize);
    if (LLVM_UNLIKELY(traceDynamicBuffer)) {
        args[3] = reportCallback;
        args[4] = pipelineHandle;
        args[5] = portNum;
    }

    b.CreateCall(f, args);
}

void ManagedDynamicBuffer::releaseBuffer(KernelBuilder & b) const {
    /* Free the dynamically allocated buffer(s). */

    StructType * handleTy = getHandleType(b);

    Module * m = b.getModule();

    auto & DL = m->getDataLayout();

    PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);
    const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();

    IntegerType * const intPtrTy = b.getIntPtrTy(DL);

    Value * const handle = getHandle();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);

    if (mLinear) {
        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * const addrField = b.CreateInBoundsGEP(handleTy, handle, indices);
        Value * const addr = b.CreateAlignedLoad(addrPtrTy, addrField, voidPtrTyAlign);
        b.CreateFree(b.CreatePointerCast(addr, b.getInt8PtrTy()));
        b.CreateAlignedStore(ConstantPointerNull::get(addrPtrTy), addrField, voidPtrTyAlign);
    } else {

        const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * const addrField = b.CreateInBoundsGEP(handleTy, handle, indices);
        Value * const addr = b.CreateAlignedLoad(addrPtrTy, addrField, voidPtrTyAlign);
        indices[1] = b.getInt32(LinearInternalCapacity);
        Value * const capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
        Value * const capacity = b.CreateAlignedLoad(intPtrTy, capacityField, intPtrTyAlign);

        FixedArray<Value *, 2> args;
        args[0] = b.CreatePointerCast(addr, b.getInt8PtrTy());
        args[1] = b.CreateMul(b.getTypeSize(mType), b.CreateShl(capacity, 1));

        Function * const fMunmap = m->getFunction(__MUNMAP); assert (fMunmap);
        b.CreateCall(fMunmap, args);

        b.CreateAlignedStore(ConstantPointerNull::get(addrPtrTy), addrField, voidPtrTyAlign);
        b.CreateAlignedStore(b.getSize(0), capacityField, intPtrTyAlign);
    }
    freePendingDeletions(b, ConstantInt::getAllOnesValue(intPtrTy));
}

Value * ManagedDynamicBuffer::getBaseAddress(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    auto & DL = b.getModule()->getDataLayout();
    PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);
    const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * field = nullptr;
    indices[1] = b.getInt32(LinearBaseAddress);
    field = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    return b.CreateAlignedLoad(addrPtrTy, field, voidPtrTyAlign, true);
}

Value * ManagedDynamicBuffer::getMallocAddress(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    auto & DL = b.getModule()->getDataLayout();
    PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);
    const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(LinearMallocedAddress);
    Value * field = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    return b.CreateAlignedLoad(addrPtrTy, field, voidPtrTyAlign, true);
}

Value * ManagedDynamicBuffer::getCapacity(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    auto & DL = b.getModule()->getDataLayout();
    IntegerType * const intPtrTy = b.getIntPtrTy(DL);
    const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    Value * field = nullptr;
    indices[1] = b.getInt32(LinearInternalCapacity);
    field = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    Value * cap = b.CreateAlignedLoad(intPtrTy, field, intPtrTyAlign, true);
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
    indices[1] = b.getInt32(LinearInternalCapacity);
    Value * field = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    Value * cap = b.CreateAlignedLoad(intPtrTy, field, intPtrTyAlign, true);
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    return b.CreateMul(BLOCK_WIDTH, cap);
}

void ManagedDynamicBuffer::setBaseAddress(KernelBuilder & b, Value * const addr) const {
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
    assert (mHandle && "has not been set prior to calling setCapacity");
    auto & DL = b.getModule()->getDataLayout();
    IntegerType * const intPtrTy = b.getIntPtrTy(DL);
    const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(LinearInternalCapacity);
    Value * ptr = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    b.CreateAlignedStore(b.CreateExactUDiv(capacity, BLOCK_WIDTH), ptr, intPtrTyAlign);
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
        indices[1] = b.getInt32(LinearInternalCapacity);
        Value * field = b.CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * cap = b.CreateAlignedLoad(intPtrTy, field, intPtrTyAlign, true);
        Value * isZero = b.CreateICmpEQ(cap, ConstantInt::getNullValue(intPtrTy));
        cap = b.CreateSelect(isZero, ConstantInt::getAllOnesValue(intPtrTy), cap);
        return b.CreateURem(offset, cap);
    }
}

static void removeFromPendingDeletions(KernelBuilder & b, Value * const pendingStruct, Value * const consumed, const bool useFree, const unsigned addrSpace) {

    Module * const m = b.getModule();
    auto & DL = m->getDataLayout();

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    assert ("unspecified module" && b.getModule());

    name << "__check_pending_deletions" << addrSpace;
    if (useFree) {
        name << "F";
    }

    Function * f = m->getFunction(name.str());

    if (f == nullptr) {

        const auto ip = b.saveIP();

        auto & C = m->getContext();

        StructType * const handleTy = makePendingDeletionStructTy(b);
        PointerType * const handlePtrTy = handleTy->getPointerTo(addrSpace);

        PointerType * const voidPtrTy = b.getVoidPtrTy();
        IntegerType * const intPtrTy = DL.getIntPtrType(C);

        const auto voidPtrTyAlign = DL.getABITypeAlign(voidPtrTy).value();
        const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

        Constant * const nilVoidPtr = ConstantPointerNull::get(voidPtrTy);

        FixedArray<Type *, 4> pendingDeletionFields;
        pendingDeletionFields[PendingDeletionAddress] = voidPtrTy;
        pendingDeletionFields[PendingDeletionCapacity] = intPtrTy;
        pendingDeletionFields[PendingDeletionConsumed] = intPtrTy;
        pendingDeletionFields[PendingDeletionNextLink] = voidPtrTy;
        StructType * const pendingDeletionTy = StructType::get(C, pendingDeletionFields);
        PointerType * const pendingDeletionPtrTy = pendingDeletionTy->getPointerTo(addrSpace);


        ConstantInt * const i32_ZERO = b.getInt32(0);
        ConstantInt * const i32_ONE = b.getInt32(1);
        ConstantInt * const i32_PendingAddress = b.getInt32(PendingDeletionAddress);
        ConstantInt * const i32_PendingCapacity = b.getInt32(PendingDeletionCapacity);
        ConstantInt * const i32_PendingConsumed = b.getInt32(PendingDeletionConsumed);
        ConstantInt * const i32_PendingNextLink = b.getInt32(PendingDeletionNextLink);

        ConstantInt * const sz_ZERO = ConstantInt::get(intPtrTy, 0);
        Constant * const sz_ALL_ONES = ConstantInt::getAllOnesValue(intPtrTy);

        FixedArray<Type *, 2> paramTypes;
        paramTypes[0] = handlePtrTy;
        paramTypes[1] = intPtrTy;
        FunctionType * const funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

        Function * const innerFunc = Function::Create(funcTy, Function::InternalLinkage, name.str() + "_I", m);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            innerFunc->setHasUWTable();
            #else
            innerFunc->setUWTableKind(UWTableKind::Default);
            #endif
        }

        BEGIN_SCOPED_REGION

        BasicBlock * const entry = BasicBlock::Create(C, "entry", innerFunc);
        BasicBlock * const removeSecondFixed = BasicBlock::Create(C, "removeSecondFixed", innerFunc);
        BasicBlock * const checkLinkedList = BasicBlock::Create(C, "checkLinkedList", innerFunc);
        BasicBlock * const deleteCurrentLink = BasicBlock::Create(C, "deleteCurrentLink", innerFunc);
        BasicBlock * const clearSecond = BasicBlock::Create(C, "clearSecond", innerFunc);
        BasicBlock * const copySecondToFirst = BasicBlock::Create(C, "copySecondToFirst", innerFunc);
        BasicBlock * const copyLinkToFixedSlots = BasicBlock::Create(C, "copyLinkToFixedSlots", innerFunc);
        BasicBlock * const checkNextLinkToPropagateToFixedSlot = BasicBlock::Create(C, "checkNextLinkToPropagateToFixedSlot", innerFunc);
        BasicBlock * const exit = BasicBlock::Create(C, "exit", innerFunc);

        b.SetInsertPoint(entry);

        auto arg = innerFunc->arg_begin();
        auto nextArg = [&]() {
            assert (arg != innerFunc->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };

        Value * const handle = nextArg();
        handle->setName("handle");
        Value * const currentlyConsumed = nextArg();
        currentlyConsumed->setName("consumed");

        FixedArray<Value *, 4> indices4;
        indices4[0] = i32_ZERO;
        indices4[1] = i32_ZERO;
        indices4[2] = i32_ZERO;
        indices4[3] = i32_PendingConsumed;

        Function * const fMunmap = m->getFunction(__MUNMAP); assert (fMunmap);

        Value * const pendingConsumed1Field = b.CreateGEP(handleTy, handle, indices4);
        indices4[3] = i32_PendingAddress;
        Value * const pendingAddr1Field = b.CreateGEP(handleTy, handle, indices4);
        Value * pendingCapacity1Field = nullptr;
        BEGIN_SCOPED_REGION
        Value * const pendingAddr1 = b.CreateAlignedLoad(voidPtrTy, pendingAddr1Field, voidPtrTyAlign);
        if (LLVM_UNLIKELY(useFree)) {
            b.CreateFree(pendingAddr1);
        } else {
            indices4[3] = i32_PendingCapacity;
            pendingCapacity1Field = b.CreateGEP(handleTy, handle, indices4);
            Value * const pendingCapacity1 = b.CreateAlignedLoad(intPtrTy, pendingCapacity1Field, intPtrTyAlign);
            FixedArray<Value *, 2> args;
            args[0] = pendingAddr1;
            args[1] = pendingCapacity1;
            b.CreateCall(fMunmap, args);
        }
        END_SCOPED_REGION
        indices4[2] = i32_ONE;
        indices4[3] = i32_PendingAddress;
        Value * const pendingAddr2Field = b.CreateGEP(handleTy, handle, indices4);
        Value * const pendingAddr2 = b.CreateAlignedLoad(voidPtrTy, pendingAddr2Field, voidPtrTyAlign);
        Value * pendingCapacity2Field = nullptr;
        Value * pendingCapacity2 = nullptr;
        if (LLVM_LIKELY(!useFree)) {
            indices4[3] = i32_PendingCapacity;
            pendingCapacity2Field = b.CreateGEP(handleTy, handle, indices4);
            pendingCapacity2 = b.CreateAlignedLoad(intPtrTy, pendingCapacity2Field, intPtrTyAlign);
        }
        FixedArray<Value *, 2> indices2;
        indices2[0] = i32_ZERO;
        indices2[1] = i32_ONE;
        Value * const additionalLinkField = b.CreateGEP(handleTy, handle, indices2);
        indices4[3] = i32_PendingConsumed;
        Value * const pendingConsumed2Field = b.CreateGEP(handleTy, handle, indices4);
        Value * const pendingConsumed2 = b.CreateAlignedLoad(intPtrTy, pendingConsumed2Field, intPtrTyAlign);
        Value * const safeToRemove2 = b.CreateICmpULT(pendingConsumed2, currentlyConsumed);
        b.CreateLikelyCondBr(safeToRemove2, removeSecondFixed, copySecondToFirst);

        // -----------------------------------------------------

        b.SetInsertPoint(copySecondToFirst);
        // We must keep the second entry (and thus any linked entries) but must delete the first one.
        // If the second entry is empty, we'll just copy a "black" entry over the first and thus clear
        // it. If this is the case, we cannot have an additional link list so we'll clear the second
        // entry. If we do have a non empty second entry, we may have a linked list and will end up
        // copying the first non deleted link list entry over the second.
        b.CreateAlignedStore(pendingAddr2, pendingAddr1Field, voidPtrTyAlign);
        if (LLVM_LIKELY(!useFree)) {
            b.CreateAlignedStore(pendingCapacity2, pendingCapacity1Field, intPtrTyAlign);
        }
        b.CreateAlignedStore(pendingConsumed2, pendingConsumed1Field, intPtrTyAlign);

        // Check if we have any additional links; we're going to have to propagate the values of the
        // first additional link into the second values.

        Value * const additionalLink = b.CreateAlignedLoad(voidPtrTy, additionalLinkField, voidPtrTyAlign);
        Value * const noAdditionalLink = b.CreateICmpEQ(additionalLink, nilVoidPtr);
        b.CreateLikelyCondBr(noAdditionalLink, clearSecond, copyLinkToFixedSlots);

        // -----------------------------------------------------

        b.SetInsertPoint(removeSecondFixed);
        if (LLVM_UNLIKELY(useFree)) {
            b.CreateFree(pendingAddr2);
        } else {
            indices4[3] = i32_PendingCapacity;
            FixedArray<Value *, 2> args;
            args[0] = pendingAddr2;
            args[1] = pendingCapacity2;
            b.CreateCall(fMunmap, args);
        }
        Value * const pendingLink = b.CreateAlignedLoad(voidPtrTy, additionalLinkField, voidPtrTyAlign);
        Value * const noPendingLink = b.CreateICmpEQ(pendingLink, nilVoidPtr);
        b.CreateLikelyCondBr(noPendingLink, clearSecond, checkLinkedList);

        // -----------------------------------------------------

        b.SetInsertPoint(checkLinkedList);
        PHINode * const currentLinkPhi = b.CreatePHI(voidPtrTy, 2);
        currentLinkPhi->addIncoming(pendingLink, removeSecondFixed);
        Value * const currentLink = b.CreatePointerCast(currentLinkPhi, pendingDeletionPtrTy);
        indices2[0] = i32_ZERO;
        indices2[1] = i32_PendingConsumed;
        Value * const pendingConsumedField = b.CreateGEP(pendingDeletionTy, currentLink, indices2);
        Value * const pendingConsumed = b.CreateAlignedLoad(intPtrTy, pendingConsumedField, intPtrTyAlign);
        Value * const safeToRemove = b.CreateICmpULT(pendingConsumed, currentlyConsumed);
        b.CreateLikelyCondBr(safeToRemove, deleteCurrentLink, copyLinkToFixedSlots);

        b.SetInsertPoint(deleteCurrentLink);
        indices2[1] = i32_PendingAddress;
        Value * const pendingAddrField = b.CreateGEP(pendingDeletionTy, currentLink, indices2);
        Value * const toDestroyAddr = b.CreateAlignedLoad(voidPtrTy, pendingAddrField, voidPtrTyAlign);

        if (LLVM_UNLIKELY(useFree)) {
            b.CreateFree(toDestroyAddr);
        } else {
            indices2[1] = i32_PendingCapacity;
            Value * const pendingCapacityField = b.CreateGEP(pendingDeletionTy, currentLink, indices2);
            Value * const pendingCapacity = b.CreateAlignedLoad(intPtrTy, pendingCapacityField, intPtrTyAlign);
            FixedArray<Value *, 2> args;
            args[0] = toDestroyAddr;
            args[1] = pendingCapacity;
            b.CreateCall(fMunmap, args);
        }

        indices2[1] = i32_PendingNextLink;
        Value * const nextLinkField = b.CreateGEP(pendingDeletionTy, currentLink, indices2);
        Value * const nextLink = b.CreateAlignedLoad(voidPtrTy, nextLinkField, voidPtrTyAlign);
        b.CreateFree(currentLinkPhi);
        currentLinkPhi->addIncoming(nextLink, deleteCurrentLink);
        Value * const noNextLink = b.CreateICmpEQ(nextLink, nilVoidPtr);
        b.CreateLikelyCondBr(noNextLink, clearSecond, checkLinkedList);

        // -----------------------------------------------------

        b.SetInsertPoint(copyLinkToFixedSlots);
        PHINode * const nextLinkPhi = b.CreatePHI(voidPtrTy, 3);
        nextLinkPhi->addIncoming(additionalLink, copySecondToFirst);
        nextLinkPhi->addIncoming(currentLinkPhi, checkLinkedList);
        PHINode * const fixedIndexPhi = b.CreatePHI(b.getInt32Ty(), 3);
        fixedIndexPhi->addIncoming(i32_ONE, copySecondToFirst);
        fixedIndexPhi->addIncoming(i32_ZERO, checkLinkedList);
        Value * const nextPendingDeletionLink = b.CreatePointerCast(nextLinkPhi, pendingDeletionPtrTy);
        indices2[1] = i32_PendingAddress;
        Value * const linkAddrField = b.CreateGEP(pendingDeletionTy, nextPendingDeletionLink, indices2);
        Value * const linkAddr = b.CreateAlignedLoad(voidPtrTy, linkAddrField, voidPtrTyAlign);
        indices4[2] = fixedIndexPhi;
        indices4[3] = i32_PendingAddress;
        Value * const pendingAddress3Field = b.CreateGEP(handleTy, handle, indices4);
        b.CreateAlignedStore(linkAddr, pendingAddress3Field, voidPtrTyAlign);

        if (!useFree) {
            indices2[1] = i32_PendingCapacity;
            Value * const linkCapacityField = b.CreateGEP(pendingDeletionTy, nextPendingDeletionLink, indices2);
            Value * const linkCapacity = b.CreateAlignedLoad(intPtrTy, linkCapacityField, intPtrTyAlign);
            indices4[3] = i32_PendingCapacity;
            Value * const pendingCapacity3Field = b.CreateGEP(handleTy, handle, indices4);
            b.CreateAlignedStore(linkCapacity, pendingCapacity3Field, intPtrTyAlign);
        }

        indices2[1] = i32_PendingConsumed;
        Value * const linkConsumedField = b.CreateGEP(pendingDeletionTy, nextPendingDeletionLink, indices2);
        Value * const linkConsumed = b.CreateAlignedLoad(intPtrTy, linkConsumedField, intPtrTyAlign);
        indices4[3] = i32_PendingConsumed;
        Value * const pendingConsumed3Field = b.CreateGEP(handleTy, handle, indices4);
        b.CreateAlignedStore(linkConsumed, pendingConsumed3Field, intPtrTyAlign);

        indices2[1] = i32_PendingNextLink;
        Value * const subsequentLinkField = b.CreateGEP(pendingDeletionTy, nextPendingDeletionLink, indices2);
        Value * const subsequentLink = b.CreateAlignedLoad(voidPtrTy, subsequentLinkField, voidPtrTyAlign);
        b.CreateAlignedStore(subsequentLink, additionalLinkField, voidPtrTyAlign);
        b.CreateFree(nextPendingDeletionLink);
        Value * const doneLinkCopy = b.CreateICmpEQ(fixedIndexPhi, i32_ONE);
        b.CreateCondBr(doneLinkCopy, exit, checkNextLinkToPropagateToFixedSlot);

        // -----------------------------------------------------

        b.SetInsertPoint(checkNextLinkToPropagateToFixedSlot);
        nextLinkPhi->addIncoming(subsequentLink, checkNextLinkToPropagateToFixedSlot);
        fixedIndexPhi->addIncoming(i32_ONE, checkNextLinkToPropagateToFixedSlot);
        Value * const noSubsequentLink = b.CreateICmpEQ(subsequentLink, nilVoidPtr);
        b.CreateCondBr(noSubsequentLink, clearSecond, copyLinkToFixedSlots);

        // -----------------------------------------------------

        b.SetInsertPoint(clearSecond);
        b.CreateAlignedStore(nilVoidPtr, pendingAddr2Field, voidPtrTyAlign);
        if (LLVM_LIKELY(!useFree)) {
            b.CreateAlignedStore(sz_ZERO, pendingCapacity2Field, intPtrTyAlign);
        }
        b.CreateAlignedStore(sz_ALL_ONES, pendingConsumed2Field, intPtrTyAlign);
        b.CreateAlignedStore(nilVoidPtr, additionalLinkField, voidPtrTyAlign);
        b.CreateBr(exit);

        // -----------------------------------------------------

        b.SetInsertPoint(exit);
        b.CreateRetVoid();

        END_SCOPED_REGION

        FunctionType * const funcTy2 = FunctionType::get(b.getVoidTy(), paramTypes, false);

        f = Function::Create(funcTy2, Function::InternalLinkage, name.str(), m);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            f->setHasUWTable();
            #else
            f->setUWTableKind(UWTableKind::Default);
            #endif
        }
        f->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);

        BasicBlock * const entry = BasicBlock::Create(C, "entry", f);
        BasicBlock * const deleteFirstPending = BasicBlock::Create(C, "deleteFirstPending", f);
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
        Value * const currentlyConsumed = nextArg();
        currentlyConsumed->setName("currentlyConsumed");

        FixedArray<Value *, 4> indices4;
        indices4[0] = i32_ZERO;
        indices4[1] = i32_ZERO;
        indices4[2] = i32_ZERO;
        indices4[3] = i32_PendingConsumed;
        Value * const pendingConsumedField = b.CreateGEP(handleTy, handle, indices4);
        Value * const pendingConsumed = b.CreateAlignedLoad(intPtrTy, pendingConsumedField, intPtrTyAlign);
        // Since the consumed item count is monotonically non decreasing, we know if they don't equal, it must be greater.
        Value * const deleteFirst = b.CreateICmpUGT(currentlyConsumed, pendingConsumed);

        b.CreateUnlikelyCondBr(deleteFirst, deleteFirstPending, exit);

        b.SetInsertPoint(deleteFirstPending);
        FixedArray<Value *, 2> args;
        args[0] = handle;
        args[1] = currentlyConsumed;
        b.CreateCall(funcTy, innerFunc, args);
        b.CreateBr(exit);

        b.SetInsertPoint(exit);
        b.CreateRetVoid();

        b.restoreIP(ip);
    }
    FixedArray<Value *, 2> args;
    args[0] = pendingStruct;
    args[1] = consumed;
    b.CreateCall(f->getFunctionType(), f, args);
}

void ManagedDynamicBuffer::freePendingDeletions(KernelBuilder & b, llvm::Value * consumed) const {
    FixedArray<Value *, 2> indices2;
    indices2[0] = b.getInt32(0);
    indices2[1] = b.getInt32(PendingDeletionStruct);
    Value * const handle = b.CreateGEP(mHandleType, mHandle, indices2);
    removeFromPendingDeletions(b, handle, consumed, mLinear, mAddressSpace);
}

static void addToPendingDeletions(KernelBuilder & b, Value * const pendingStruct,
                                  Value * const addressToDelete, Value * capacity, Value * bytesPerChunk, Value * const safeToDeleteAt,
                                  const size_t addrSpace) {

    Module * const m = b.getModule();

    Function * f = m->getFunction("__addPendingStreamSetDeletion");

    if (f == nullptr) {

        const auto ip = b.saveIP();

        auto & C = b.getContext();

        auto & DL = m->getDataLayout();

        PointerType * const voidPtrTy = b.getVoidPtrTy();
        const auto voidPtrTyAlign = DL.getABITypeAlign(voidPtrTy).value();

        IntegerType * const intPtrTy = DL.getIntPtrType(C);
        const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

        StructType * const handleTy = makePendingDeletionStructTy(b);
        PointerType * const handlePtrTy = handleTy->getPointerTo();

        FixedArray<Type *, 4> paramTypes;
        paramTypes[0] = handlePtrTy; // pending struct ptr
        paramTypes[1] = voidPtrTy; // address to delete
        paramTypes[2] = intPtrTy; // capacity (bytes)
        paramTypes[3] = intPtrTy; // safe to delete at item count

        FunctionType * funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

        f = Function::Create(funcTy, Function::InternalLinkage, "__addPendingStreamSetDeletion", m);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            f->setHasUWTable();
            #else
            f->setUWTableKind(UWTableKind::Default);
            #endif
        }

        BasicBlock * const entry = BasicBlock::Create(C, "entry", f);
        BasicBlock * const entry0Empty = BasicBlock::Create(C, "entry0Empty", f);
        BasicBlock * const entry0Used = BasicBlock::Create(C, "entry0Used", f);
        BasicBlock * const entry1Empty = BasicBlock::Create(C, "entry1Empty", f);
        BasicBlock * const entry1Used = BasicBlock::Create(C, "entry1Used", f);
        BasicBlock * const scanLinkedList = BasicBlock::Create(C, "scanLinkedList", f);
        BasicBlock * const appendLinkedList = BasicBlock::Create(C, "appendLinkedList", f);
        BasicBlock * const exit = BasicBlock::Create(C, "exit", f);

        ConstantInt * const i32_PendingDeletionAddress = b.getInt32(PendingDeletionAddress);
        ConstantInt * const i32_PendingDeletionCapacity = b.getInt32(PendingDeletionCapacity);
        ConstantInt * const i32_PendingDeletionConsumed = b.getInt32(PendingDeletionConsumed);
        ConstantInt * const i32_PendingDeletionNextLink = b.getInt32(PendingDeletionNextLink);

        ConstantInt * const i32_ZERO = b.getInt32(0);
        ConstantInt * const i32_ONE = b.getInt32(1);

        Constant * const nilVoidPtr = ConstantPointerNull::get(voidPtrTy);

        PointerType * const voidPtrPtrTy = voidPtrTy->getPointerTo(addrSpace);


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
        Value * const addr = nextArg();
        addr->setName("addr");
        Value * const size = nextArg();
        size->setName("size");
        Value * const safeToDeleteAt = nextArg();
        safeToDeleteAt->setName("safeToDeleteAt");
        assert (arg == f->arg_end());

        FixedArray<Value *, 4> indices4;
        indices4[0] = i32_ZERO;
        indices4[1] = i32_ZERO;
        indices4[2] = i32_ZERO;
        indices4[3] = i32_PendingDeletionAddress;
        Value * const addr0Field = b.CreateGEP(handleTy, handle, indices4);
        Value * const addr0 = b.CreateAlignedLoad(voidPtrTy, addr0Field, voidPtrTyAlign);
        Value * const nil0 = b.CreateICmpEQ(addr0, nilVoidPtr);
        b.CreateLikelyCondBr(nil0, entry0Empty, entry0Used);

        b.SetInsertPoint(entry0Empty);
        b.CreateAlignedStore(addr, addr0Field, voidPtrTyAlign);

        indices4[3] = i32_PendingDeletionCapacity;
        Value * const cap0Field = b.CreateGEP(handleTy, handle, indices4);
        b.CreateAlignedStore(size, cap0Field, intPtrTyAlign);

        indices4[3] = i32_PendingDeletionConsumed;
        Value * const consumed0Field = b.CreateGEP(handleTy, handle, indices4);
        b.CreateAlignedStore(safeToDeleteAt, consumed0Field, intPtrTyAlign);
        b.CreateBr(exit);

        b.SetInsertPoint(entry0Used);
        indices4[2] = i32_ONE;
        indices4[3] = i32_PendingDeletionAddress;
        Value * const addr1Field = b.CreateGEP(handleTy, handle, indices4);
        Value * const addr1 = b.CreateAlignedLoad(voidPtrTy, addr1Field, voidPtrTyAlign);
        Value * const nil1 = b.CreateICmpEQ(addr1, nilVoidPtr);
        b.CreateLikelyCondBr(nil1, entry1Empty, entry1Used);

        b.SetInsertPoint(entry1Empty);
        b.CreateAlignedStore(addr, addr1Field, voidPtrTyAlign);
        indices4[3] = i32_PendingDeletionCapacity;
        Value * const cap1Field = b.CreateGEP(handleTy, handle, indices4);
        b.CreateAlignedStore(size, cap1Field, intPtrTyAlign);
        indices4[3] = i32_PendingDeletionConsumed;
        Value * const consumed1Field = b.CreateGEP(handleTy, handle, indices4);
        b.CreateAlignedStore(safeToDeleteAt, consumed1Field, intPtrTyAlign);
        b.CreateBr(exit);

        b.SetInsertPoint(entry1Used);
        FixedArray<Value *, 2> indices2;
        indices2[0] = i32_ZERO;
        indices2[1] = i32_ONE;
        Value * const initialAdditionalStructPtrField = b.CreateGEP(handleTy, handle, indices2);
        Value * const voidPtrStructPtr = b.CreatePointerCast(initialAdditionalStructPtrField, voidPtrPtrTy);
        Value * const firstLinkAddr = b.CreateAlignedLoad(voidPtrTy, initialAdditionalStructPtrField, voidPtrTyAlign);
        Value * const emptyFirstLink = b.CreateICmpEQ(firstLinkAddr, nilVoidPtr);
        b.CreateLikelyCondBr(emptyFirstLink, appendLinkedList, scanLinkedList);

        b.SetInsertPoint(scanLinkedList);
        PHINode * const linkPtrPhi = b.CreatePHI(voidPtrTy, 2);
        linkPtrPhi->addIncoming(firstLinkAddr, entry1Used);

        FixedArray<Type *, 4> pendingDeletionFields;
        pendingDeletionFields[PendingDeletionAddress] = voidPtrTy;
        pendingDeletionFields[PendingDeletionCapacity] = intPtrTy;
        pendingDeletionFields[PendingDeletionConsumed] = intPtrTy;
        pendingDeletionFields[PendingDeletionNextLink] = voidPtrTy;
        StructType * const pendingDeletionTy = StructType::get(C, pendingDeletionFields);
        PointerType * const pendingDeletionPtrTy = pendingDeletionTy->getPointerTo(addrSpace);

        indices2[0] = i32_ZERO;
        indices2[1] = i32_PendingDeletionNextLink;

        Value * const linkPtr = b.CreatePointerCast(linkPtrPhi, pendingDeletionPtrTy);
        Value * const nextLinkField = b.CreateGEP(pendingDeletionTy, linkPtr, indices2);
        Value * const voidPtrNextLinkField = b.CreatePointerCast(nextLinkField, voidPtrPtrTy);
        Value * const currentLinkAddr = b.CreateAlignedLoad(voidPtrTy, nextLinkField, voidPtrTyAlign);

        linkPtrPhi->addIncoming(currentLinkAddr, scanLinkedList);

        Value * const emptyLink = b.CreateICmpEQ(currentLinkAddr, nilVoidPtr);
        b.CreateLikelyCondBr(emptyLink, appendLinkedList, scanLinkedList);

        // we filled both the fixed slots; append another entry as a linked list node.

        b.SetInsertPoint(appendLinkedList);
        PHINode * const storeLinkPtrPhi = b.CreatePHI(voidPtrPtrTy, 2);
        storeLinkPtrPhi->addIncoming(voidPtrStructPtr, entry1Used);
        storeLinkPtrPhi->addIncoming(voidPtrNextLinkField, scanLinkedList);

        const auto linkNodeSize = b.getTypeSize(DL, pendingDeletionTy);
        const auto linkNodeAlign = b.getAlignOf(DL, pendingDeletionTy);
        Value * const newLink = b.CreateAlignedMalloc(b.getSize(linkNodeSize), linkNodeAlign);
        indices2[0] = i32_ZERO;
        indices2[1] = i32_PendingDeletionAddress;
        Value * const link = b.CreatePointerCast(newLink, pendingDeletionPtrTy);
        Value * const newAddrField = b.CreateGEP(pendingDeletionTy, link, indices2);
        b.CreateAlignedStore(addr, newAddrField, voidPtrTyAlign);

        indices2[1] = i32_PendingDeletionCapacity;
        Value * const newCapField = b.CreateGEP(pendingDeletionTy, link, indices2);
        b.CreateAlignedStore(size, newCapField, intPtrTyAlign);

        indices2[1] = i32_PendingDeletionConsumed;
        Value * const newConsumedField = b.CreateGEP(pendingDeletionTy, link, indices2);
        b.CreateAlignedStore(safeToDeleteAt, newConsumedField, intPtrTyAlign);

        indices2[1] = i32_PendingDeletionNextLink;
        Value * const newNextLink = b.CreateGEP(pendingDeletionTy, link, indices2);
        b.CreateAlignedStore(nilVoidPtr, newNextLink, voidPtrTyAlign);

        b.CreateAlignedStore(b.CreatePointerCast(newLink, voidPtrTy), storeLinkPtrPhi, voidPtrTyAlign);
        b.CreateBr(exit);

        b.SetInsertPoint(exit);
        b.CreateRetVoid();

        b.restoreIP(ip);
    }

    FixedArray<Value *, 4> args;
    args[0] = pendingStruct;
    args[1] = addressToDelete;
    args[2] = b.CreateMul(capacity, bytesPerChunk);
    args[3] = safeToDeleteAt;
    b.CreateCall(f->getFunctionType(), f, args);

}

Value * ManagedDynamicBuffer::reserveCapacity(KernelBuilder & b, Value * produced, Value * consumed, Value * required, Value * reportCallback, Value * pipelineHandle, Value * portNum) const {

    // TODO: if we keep the buffer fd, manual indicates we can resize it by calling ftruncate again. How would this
    // affect users of the buffer prior to setting the new addr/capacity? The mmap'ed buffer pointing to the fds
    // have a set size that may mean its a non-issue as long as ftruncating the buffer doesn't invalidate them.

    Module * const m = b.getModule();
    auto & DL = m->getDataLayout();

    const auto traceDynamicBuffer = (reportCallback != nullptr);

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    assert ("unspecified module" && b.getModule());

    name << "__ManagedDynamicBuffer";
    if (mLinear) {
        name << "Linear";
    }
    name << "_reserve_capacity" << mAddressSpace;
    if (LLVM_UNLIKELY(traceDynamicBuffer)) {
        name << 'T';
    }

    PointerType * const voidPtrTy = b.getVoidPtrTy();
    const auto blockWidth = b.getBitBlockWidth();

    ConstantInt * const sz_ONE = b.getSize(1);

    Function * f = m->getFunction(name.str());

    if (f == nullptr) {

        const auto ip = b.saveIP();

        StructType * const handleTy = getHandleType(b);
        PointerType * const handlePtrTy = handleTy->getPointerTo(mAddressSpace);

        PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);

        IntegerType * const i8Ty = b.getInt8Ty();

        PointerType * const i8PtrTy = b.getInt8PtrTy();
        const auto voidPtrTyAlign = DL.getABITypeAlign(i8PtrTy).value();

        FixedVectorType * const vecTy = b.getBitBlockType();

        Constant * const log2BytesPerBlock = b.getSize(floor_log2(b.getTypeSize(DL, vecTy)));

        auto & C = b.getContext();
        IntegerType * const intPtrTy = DL.getIntPtrType(C);
        const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

        ConstantInt * const BLOCK_WIDTH = b.getSize(blockWidth);
        ConstantInt * const sz_ZERO = b.getSize(0);

        ConstantInt * const i32_ZERO = b.getInt32(0);

        constexpr auto DUFF_STEPS = 8;

        SmallVector<Type *, 8> paramTypes(traceDynamicBuffer ? 8 : 5);
        paramTypes[0] = voidPtrTy; // shared struct ptr
        paramTypes[1] = intPtrTy; // produced
        paramTypes[2] = intPtrTy; // consumed
        paramTypes[3] = intPtrTy; // required
        paramTypes[4] = intPtrTy; // blocksPerChunk
        if (LLVM_UNLIKELY(traceDynamicBuffer)) {
            paramTypes[5] = voidPtrTy; // reportExpansionCallback
            paramTypes[6] = voidPtrTy; // pipelineHandle
            paramTypes[7] = intPtrTy; // portNum
        }

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
        BasicBlock * const copyEntry = BasicBlock::Create(C, "copyEntry", f);
        BasicBlock * const copyExit = BasicBlock::Create(C, "copyExit", f);
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
        Value * const produced = nextArg();
        produced->setName("produced");
        Value * const consumed = nextArg();
        consumed->setName("consumed");
        Value * const required = nextArg();
        required->setName("required");
        Value * const bytesPerChunk = nextArg();
        bytesPerChunk->setName("bytesPerChunk");
        Value * reportExpansionCallback = nullptr;
        Value * pipelineHandle = nullptr;
        Value * portNum = nullptr;
        if (LLVM_UNLIKELY(traceDynamicBuffer)) {
            reportExpansionCallback = nextArg();
            reportExpansionCallback->setName("reportExpansionCallback");
            pipelineHandle = nextArg();
            pipelineHandle->setName("pipelineHandle");
            portNum = nextArg();
            portNum->setName("portNum");
        }
        assert (arg == f->arg_end());

        Value * const consumedChunks = b.CreateUDiv(consumed, BLOCK_WIDTH);
        Value * const requiredChunks = b.CreateCeilUDiv(b.CreateAdd(produced, required), BLOCK_WIDTH);

        FixedArray<Value *, 2> indices;
        indices[0] = i32_ZERO;
        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * const addrField = b.CreateInBoundsGEP(handleTy, handle, indices);
        Value * const initialAddr = b.CreateAlignedLoad(i8PtrTy, addrField, voidPtrTyAlign);
        indices[1] = b.getInt32(LinearInternalCapacity);
        Value * const capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
        Value * const initialCapacity = b.CreateAlignedLoad(intPtrTy, capacityField, intPtrTyAlign);

        assert (blockWidth > 8);

        BasicBlock * expandInternalBuffer = nullptr;
        BasicBlock * reuseExistingBuffer = nullptr;
        Value * virtualBaseAddrField = nullptr;
        Value * startOfUnreadBuffer = nullptr;
        Value * cannotReuseCurrentBuffer = nullptr;
        Value * const pendingChunks = b.CreateSub(requiredChunks, consumedChunks);

        if (mLinear) {
            expandInternalBuffer = BasicBlock::Create(C, "expandAndCopyBack", f, copyEntry);
            reuseExistingBuffer = BasicBlock::Create(C, "reuseExistingBuffer", f, copyEntry);
            indices[1] = b.getInt32(LinearBaseAddress);
            virtualBaseAddrField = b.CreateInBoundsGEP(handleTy, handle, indices);
            Value * virtualBase = b.CreateAlignedLoad(i8PtrTy, virtualBaseAddrField, voidPtrTyAlign);
            startOfUnreadBuffer = b.CreateInBoundsGEP(i8Ty, virtualBase, b.CreateMul(consumedChunks, bytesPerChunk));
            Value * intStartOfUnreadBuffer = b.CreatePtrToInt(startOfUnreadBuffer, intPtrTy);
            Value * requiresUpToPosition = b.CreateInBoundsGEP(i8Ty, initialAddr, b.CreateMul(pendingChunks, bytesPerChunk));
            Value * intRequiresUpToPosition = b.CreatePtrToInt(requiresUpToPosition, intPtrTy);
            cannotReuseCurrentBuffer = b.CreateICmpUGT(intRequiresUpToPosition, intStartOfUnreadBuffer);
            b.CreateUnlikelyCondBr(cannotReuseCurrentBuffer, expandInternalBuffer, reuseExistingBuffer);

            b.SetInsertPoint(expandInternalBuffer);
        }

        // allocate a buffer with at least twice the capacity and copy the unconsumed data back into it
        Value * expandedCapacity = b.CreateAdd(initialCapacity, b.CreateRoundUp(pendingChunks, initialCapacity));
        Value * const expandedBytes = b.CreateMul(expandedCapacity, bytesPerChunk);
        Value * expandedAddr = nullptr;

        if (mLinear) {
            expandedAddr = b.CreatePageAlignedMalloc(expandedBytes);

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
        } else {
            FixedArray<Value *, 2> makeArgs;
            makeArgs[0] = expandedBytes;
            makeArgs[1] = sz_ZERO;
            Function * const makeBuffer = m->getFunction(__MAKE_CIRCULAR_BUFFER); assert (makeBuffer);
            expandedAddr = b.CreatePointerCast(b.CreateCall(makeBuffer, makeArgs), i8PtrTy);
        }

        // copy the data over to the new/reused buffer
        Value * const producedChunks = b.CreateCeilUDiv(produced, BLOCK_WIDTH);
        Value * const unreadChunks = b.CreateSub(producedChunks, consumedChunks);
        b.CreateCondBr(b.CreateICmpEQ(unreadChunks, sz_ZERO), copyExit, copyEntry);

        b.SetInsertPoint(copyEntry);
        Value * toCopyPtr = nullptr;
        Value * unreadDataPtr = nullptr;
        if (mLinear) {
            toCopyPtr = expandedAddr; assert (expandedAddr);
            unreadDataPtr = startOfUnreadBuffer; assert (startOfUnreadBuffer);
        } else {
            Value * const newBufferOffset = b.CreateURem(consumedChunks, expandedCapacity);
            toCopyPtr = b.CreateInBoundsGEP(i8Ty, expandedAddr, b.CreateMul(newBufferOffset, bytesPerChunk));
            Value * const oldBufferOffset = b.CreateURem(consumedChunks, initialCapacity);
            unreadDataPtr = b.CreateInBoundsGEP(i8Ty, initialAddr, b.CreateMul(oldBufferOffset, bytesPerChunk));
        }

        if (LLVM_LIKELY(b.supportsIndirectBr())) {

            FixedArray<BasicBlock *, DUFF_STEPS> copyLoopEntryPoint;
            for (unsigned i = 0; i < DUFF_STEPS; ++i) {
                copyLoopEntryPoint[i] = BasicBlock::Create(C, "copyLoopEntryPoint", f, copyExit);
            }

            FixedArray<Constant *, DUFF_STEPS> copyLoopAddr;
            for (unsigned i = 0; i < DUFF_STEPS; ++i) {
                copyLoopAddr[i] = BlockAddress::get(f, copyLoopEntryPoint[(DUFF_STEPS - i) % DUFF_STEPS]);
            }

            ArrayType * const copyLoopArrayAddrTy = ArrayType::get(i8PtrTy, DUFF_STEPS);
            Constant * const copyLoopAddrArray = ConstantArray::get(copyLoopArrayAddrTy, copyLoopAddr);

            GlobalVariable * const copyLoopTargetArray =
                new GlobalVariable(*m, copyLoopArrayAddrTy, true, GlobalValue::ExternalLinkage, copyLoopAddrArray);

            PointerType * const vecPtrTy = vecTy->getPointerTo(mAddressSpace);

            const auto vecAlign = DL.getABITypeAlign(vecTy).value();

            const auto i8PtrAlign = DL.getABITypeAlign(i8PtrTy).value();

            Value * const blocksPerChunk = b.CreateLShr(bytesPerChunk, log2BytesPerBlock);

            Value * const maxNumOfBlocks = b.CreateMul(unreadChunks, blocksPerChunk);

            FixedArray<Value *, 2> jumpIndex;
            jumpIndex[0] = sz_ZERO;
            jumpIndex[1] = b.CreateURemRational(maxNumOfBlocks, DUFF_STEPS);
            unreadDataPtr = b.CreatePointerCast(unreadDataPtr, vecPtrTy);
            toCopyPtr = b.CreatePointerCast(toCopyPtr, vecPtrTy);

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
                Value * inputPtr = b.CreateGEP(vecTy, unreadDataPtr, copyAddrIndexPhi[i]);
                Value * outputPtr = b.CreateGEP(vecTy, toCopyPtr, copyAddrIndexPhi[i]);
                Value * in = b.CreateAlignedLoad(vecTy, inputPtr, vecAlign);
                b.CreateAlignedStore(in, outputPtr, vecAlign);
                Value * next = b.CreateAdd(copyAddrIndexPhi[i], sz_ONE);
                const auto k = (i + 1U) % DUFF_STEPS;
                copyAddrIndexPhi[k]->addIncoming(next, copyLoopEntryPoint[i]);
                if (k == 0) {
                    Value * const isDone = b.CreateICmpEQ(next, maxNumOfBlocks);
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
        b.CreateAlignedStore(expandedAddr, addrField, voidPtrTyAlign);
        b.CreateAlignedStore(expandedCapacity, capacityField, intPtrTyAlign);
        if (mLinear) {
            Value * consumedOffset =  b.CreateNeg(b.CreateMul(consumedChunks, bytesPerChunk));
            Value * const newVirtualAddress = b.CreateInBoundsGEP(b.getInt8Ty(), expandedAddr, consumedOffset);
            b.CreateAlignedStore(b.CreatePointerCast(newVirtualAddress, addrPtrTy), virtualBaseAddrField, voidPtrTyAlign);
        } else {
            indices[1] = b.getInt32(LinearBaseAddress);
            virtualBaseAddrField = b.CreateInBoundsGEP(handleTy, handle, indices);
            b.CreateAlignedStore(expandedAddr, virtualBaseAddrField, voidPtrTyAlign);
        }

//        Value * const effectiveCapacity = b.CreateAdd(consumedChunks, expandedCapacity);
//        indices[1] = b.getInt32(LinearEffectiveCapacity);
//        Value * baseCapacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
//        b.CreateAlignedStore(effectiveCapacity, baseCapacityField, intPtrTyAlign);

        BasicBlock * addPendingDeletion = nullptr;
        if (mLinear) {
            addPendingDeletion = BasicBlock::Create(C, "addPendingDeletion", f, exit);

            b.CreateUnlikelyCondBr(cannotReuseCurrentBuffer, addPendingDeletion, exit);

            b.SetInsertPoint(addPendingDeletion);
        }

        if (LLVM_UNLIKELY(traceDynamicBuffer)) {

            FixedArray<Type *, 4> paramTypes;
            paramTypes[0] = voidPtrTy; // pipeline handle
            paramTypes[1] = intPtrTy; // port num
            paramTypes[2] = intPtrTy; // produced
            paramTypes[3] = intPtrTy; // capacity

            FunctionType * funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

            FixedArray<Value *, 4> callbackArgs;
            callbackArgs[0] = pipelineHandle;
            callbackArgs[1] = portNum;
            callbackArgs[2] = produced;
            callbackArgs[3] = b.CreateMul(expandedCapacity, BLOCK_WIDTH);

            b.CreateCall(funcTy, b.CreatePointerCast(reportExpansionCallback, funcTy->getPointerTo()), callbackArgs);

        }

        FixedArray<Value *, 2> indices2;
        indices2[0] = i32_ZERO;
        indices2[1] = b.getInt32(PendingDeletionStruct);
        Value * const pendingField = b.CreateGEP(handleTy, handle, indices2);
        Value * const safeToDeleteAt = b.CreateAdd(produced, BLOCK_WIDTH);
        addToPendingDeletions(b, pendingField, initialAddr, initialCapacity, b.CreateShl(bytesPerChunk, 1), safeToDeleteAt, mAddressSpace);

        b.CreateBr(exit);

        b.SetInsertPoint(exit);
        if (mLinear) {
            // TODO: return 2 if we "reuse" the data but do not copy anything?
            b.CreateRet(b.CreateZExt(cannotReuseCurrentBuffer, intPtrTy));
        } else {
            b.CreateRetVoid();
        }

        b.restoreIP(ip);

    }

    SmallVector<Value *, 8> args(traceDynamicBuffer ? 8 : 5);
    args[0] = b.CreatePointerCast(mHandle, voidPtrTy);
    args[1] = produced;
    args[2] = consumed;
    args[3] = required;
    args[4] = b.getSize(b.getTypeSize(DL, mType));
    if (LLVM_UNLIKELY(traceDynamicBuffer)) {
        args[5] = reportCallback; // reportExpansionCallback
        args[6] = pipelineHandle; // pipelineHandle
        args[7] = portNum; // portNum
    }
    Value * const retVal = b.CreateCall(f, args);
    if (mLinear) {
        return retVal;
    } else {
        return nullptr;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isItemAlignedAccessWithinStreamSetMemory
 ** ------------------------------------------------------------------------------------------------------------- */
void ManagedDynamicBuffer::assertAccessIsWithinStreamSetMemory(KernelBuilder & b, Constant * name, Value * ptr, const size_t size, Value * const start, Value * const end) const {

    assert (codegen::DebugOptionIsSet(codegen::EnableStreamSetAsserts, codegen::EnableAsserts));

    ConstantInt * sz_ZERO = b.getSize(0);
    ConstantInt * sz_LOG_2_BW = b.getSize(floor_log2(b.getBitBlockWidth()));

    Value * const ba = getBaseAddress(b);

    Value * const startIndex = b.CreateLShr(start, sz_LOG_2_BW);

    Value * startPtr = getStreamBlockPtr(b, ba, sz_ZERO, startIndex);

    Value * const endIndex = b.CreateLShr(end, sz_LOG_2_BW);

    Value * endPtr = nullptr;
    if (mLinear) {
        endPtr = getStreamBlockPtr(b, ba, sz_ZERO, endIndex);
    } else {
        endPtr = StreamSetBuffer::getStreamBlockPtr(b, b.CreatePointerCast(startPtr, ba->getType()), sz_ZERO, b.CreateSub(endIndex, startIndex));
    }

    auto & dl = b.getModule()->getDataLayout();
    IntegerType * const intPtrTy = dl.getIntPtrType(b.getContext());
    startPtr = b.CreatePtrToInt(startPtr, intPtrTy);
    Value * ptrInt = b.CreatePtrToInt(ptr, intPtrTy);
    Value * valid = b.CreateICmpULE(startPtr, ptrInt);
    Value * outPtr = b.CreateAdd(ptrInt, ConstantInt::get(intPtrTy, size)); assert (size > 0);
    endPtr = b.CreatePtrToInt(endPtr, intPtrTy);
    valid = b.CreateAnd(valid, b.CreateICmpULE(outPtr, endPtr));
    b.CreateAssert(valid, "streamset \"%s\" memory access [%" PRIx64 ",%" PRIx64 ") is outside of valid memory boundaries [%" PRIx64 ",%" PRIx64 ")",
                   name, ptr, outPtr, startPtr, endPtr);

}

// Fd-Backed Dynamic Buffer

StructType * FdBackedDynamicBuffer::getHandleType(KernelBuilder & b) const {
    if (mHandleType == nullptr) {
        auto & C = b.getContext();
        FixedArray<Type *, 4> fields;
        fields[FDDB_Field::LinearMallocedAddress] = getPointerType();
        fields[FDDB_Field::LinearCapacity] = b.getSizeTy();
        fields[FDDB_Field::PendingDeletionStruct] = makePendingDeletionStructTy(b);
        fields[FDDB_Field::Fd] = b.getIntNTy(8 * sizeof(int));
        mHandleType =  StructType::get(C, fields);
    }
    return mHandleType;
}

void FdBackedDynamicBuffer::allocateBuffer(KernelBuilder & b, Value * const capacityMultiplier, Value * reportCallback, Value * pipelineHandle, Value * portNum) {
    assert (mHandle && "has not been set prior to calling allocateBuffer");

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    assert ("unspecified module" && b.getModule());
    assert (mLinear);

    const auto traceDynamicBuffer = (reportCallback != nullptr);

    name << "__FdBackedDynamicBuffer_initial_alloc" << mAddressSpace;
    if (LLVM_UNLIKELY(traceDynamicBuffer)) {
        name << 'T';
    }

    Module * const m = b.getModule();

    auto & DL = m->getDataLayout();

    Function * f = m->getFunction(name.str());

    Type * const voidPtrTy = b.getVoidPtrTy();

    if (f == nullptr) {

        LLVMContext & C = m->getContext();

        StructType * handleTy = getHandleType(b);
        PointerType * const handlePtrTy = handleTy->getPointerTo(mAddressSpace);

        PointerType * const addrPtrTy = b.getVoidPtrTy();
        IntegerType * const intPtrTy = b.getIntPtrTy(DL);

        const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();
        const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

        SmallVector<Type *, 6> paramTypes(traceDynamicBuffer ? 6 : 3);
        paramTypes[0] = voidPtrTy;
        paramTypes[1] = intPtrTy;
        paramTypes[2] = intPtrTy;
        if (LLVM_UNLIKELY(traceDynamicBuffer)) {
            paramTypes[3] = voidPtrTy;
            paramTypes[4] = voidPtrTy;
            paramTypes[5] = intPtrTy;
        }

        FunctionType * funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

        const auto ip = b.saveIP();

        f = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            f->setHasUWTable();
            #else
            f->setUWTableKind(UWTableKind::Default);
            #endif
        }

        BasicBlock * const entry = BasicBlock::Create(C, "entry", f);
        BasicBlock * const exit = BasicBlock::Create(C, "exit", f);

        b.SetInsertPoint(entry);

        auto arg = f->arg_begin();
        auto nextArg = [&]() {
            assert (arg != f->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };

        Value * handle = nextArg();
        handle->setName("handle");
        handle = b.CreatePointerCast(handle, handlePtrTy);
        Value * capacity = nextArg();
        capacity->setName("capacity");
        Value * typeSize = nextArg();
        typeSize->setName("typeSize");
        Value * reportCallback = nullptr;
        Value * pipelineHandle = nullptr;
        Value * portNum = nullptr;
        if (LLVM_UNLIKELY(traceDynamicBuffer)) {
            reportCallback = nextArg();
            reportCallback->setName("reportCallback");
            pipelineHandle = nextArg();
            pipelineHandle->setName("pipelineHandle");
            portNum = nextArg();
            portNum->setName("portNum");
        }
        assert (arg == f->arg_end());

        ConstantInt * const i32_ZERO = b.getInt32(0);
        ConstantInt * const i32_ONE = b.getInt32(1);

        Value * capacityBytes = b.CreateMul(typeSize, capacity);

        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            b.CreateAssert(capacity, "capacity cannot be 0.");
            b.CreateAssertZero(b.CreateURemRational(capacityBytes, getPageSize()), "%" PRIu64 " x %" PRIu64, typeSize, capacity);
        }


        FixedArray<Value *, 2> indices;
        indices[0] = i32_ZERO;
        indices[1] = b.getInt32(FDDB_Field::Fd);
        Value * const fdField = b.CreateInBoundsGEP(handleTy, handle, indices);

        Function * const makeFdBackedBuffer = m->getFunction(__MAKE_FD_BACKED_BUFFER);  assert (makeFdBackedBuffer);

        FixedArray<Value *, 2> args;
        args[0] = capacityBytes;
        args[1] = fdField;
        Value * const addr = b.CreateCall(makeFdBackedBuffer->getFunctionType(), makeFdBackedBuffer, args);

        indices[1] = b.getInt32(FDDB_Field::LinearMallocedAddress);
        Value * const addrField = b.CreateInBoundsGEP(handleTy, handle, indices);
        b.CreateAlignedStore(addr, addrField, voidPtrTyAlign);
        indices[1] = b.getInt32(FDDB_Field::LinearCapacity);
        Value * const capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
        b.CreateAlignedStore(capacity, capacityField, voidPtrTyAlign);

        b.CreateBr(exit);

        // --------------------------------------------------------

        b.SetInsertPoint(exit);

        FixedArray<Value *, 5> indices5;
        indices5[0] = i32_ZERO;
        indices5[1] = b.getInt32(PendingDeletionStruct);
        indices5[2] = i32_ZERO;
        indices5[3] = i32_ZERO;
        indices5[4] = b.getInt32(PendingDeletionConsumed);

        Constant * const sz_ALL_ONES = ConstantInt::getAllOnesValue(intPtrTy);

        Value * const consumed0Field = b.CreateGEP(handleTy, handle, indices5);
        b.CreateAlignedStore(sz_ALL_ONES, consumed0Field, intPtrTyAlign);

        indices5[3] = i32_ONE;

        Value * const consumed1Field = b.CreateGEP(handleTy, handle, indices5);
        b.CreateAlignedStore(sz_ALL_ONES, consumed1Field, intPtrTyAlign);

        if (LLVM_UNLIKELY(traceDynamicBuffer)) {

            FixedArray<Type *, 4> paramTypes;
            paramTypes[0] = voidPtrTy; // pipeline handle
            paramTypes[1] = intPtrTy; // port num
            paramTypes[2] = intPtrTy; // produced
            paramTypes[3] = intPtrTy; // capacity

            FunctionType * funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

            FixedArray<Value *, 4> callbackArgs;
            callbackArgs[0] = pipelineHandle;
            callbackArgs[1] = portNum;
            callbackArgs[2] = b.getSize(0);
            callbackArgs[3] = b.CreateMul(capacity, b.getSize(b.getBitBlockWidth()));

            b.CreateCall(funcTy, b.CreatePointerCast(reportCallback, funcTy->getPointerTo()), callbackArgs);
        }

        b.CreateRetVoid();

        b.restoreIP(ip);
    }

    const auto typeSize = b.getTypeSize(DL, mType);
    assert (typeSize > 0);
    assert ((typeSize % (b.getBitBlockWidth() / 8)) == 0);

    // Let C be the capacity multiplyer, T be the type size and P be the page size.
    // We want to solve (C * T) mod P = 0, where T and P are constants but C isn't
    // known until after JIT-compilation.

    Rational stridesPerPage{getPageSize(), typeSize};
    Value * capacity = b.CreateRoundUpRational(capacityMultiplier, stridesPerPage.numerator());
    SmallVector<Value *, 6> args(traceDynamicBuffer ? 6 : 3);
    args[0] = b.CreatePointerCast(getHandle(), voidPtrTy);
    args[1] = capacity;
    args[2] = b.getSize(typeSize);
    if (LLVM_UNLIKELY(traceDynamicBuffer)) {
        args[3] = reportCallback;
        args[4] = pipelineHandle;
        args[5] = portNum;
    }
    b.CreateCall(f, args);
}

void FdBackedDynamicBuffer::releaseBuffer(KernelBuilder & b) const {

    /* Free the dynamically allocated buffer(s). */

    StructType * handleTy = getHandleType(b);

    Module * m = b.getModule();

    auto & DL = m->getDataLayout();

    IntegerType * const intTy = b.getIntNTy(8 * sizeof(int));

    PointerType * const addrPtrTy = mType->getPointerTo(mAddressSpace);
    const auto voidPtrTyAlign = DL.getABITypeAlign(addrPtrTy).value();

    IntegerType * const intPtrTy = b.getIntPtrTy(DL);

    Value * const handle = getHandle();
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);

    const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

    indices[1] = b.getInt32(FDDB_Field::LinearMallocedAddress);
    Value * const addrField = b.CreateInBoundsGEP(handleTy, handle, indices);
    Value * const addr = b.CreateAlignedLoad(addrPtrTy, addrField, voidPtrTyAlign);
    indices[1] = b.getInt32(FDDB_Field::LinearCapacity);
    Value * const capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
    Value * const capacity = b.CreateAlignedLoad(intPtrTy, capacityField, intPtrTyAlign);

    FixedArray<Value *, 2> args2;
    args2[0] = b.CreatePointerCast(addr, b.getInt8PtrTy());
    args2[1] = b.CreateMul(b.getTypeSize(mType), capacity);
    b.CreateCall(m->getFunction(__MUNMAP), args2);

    b.CreateAlignedStore(ConstantPointerNull::get(addrPtrTy), addrField, voidPtrTyAlign);
    b.CreateAlignedStore(b.getSize(0), capacityField, intPtrTyAlign);

    indices[1] = b.getInt32(FDDB_Field::Fd);
    Value * const fdField = b.CreateInBoundsGEP(handleTy, handle, indices);
    const auto intTyAlign = DL.getABITypeAlign(intTy).value();
    FixedArray<Value *, 1> args1;
    args1[0] = b.CreateAlignedLoad(intTy, fdField, intTyAlign);
    Function * fClose = m->getFunction(__CLOSE); assert (fClose);
    b.CreateCall(fClose, args1);

    freePendingDeletions(b, ConstantInt::getAllOnesValue(intPtrTy));
}

Value * FdBackedDynamicBuffer::reserveCapacity(KernelBuilder & b, Value * produced, Value * /* consumed */, Value * const required, Value * reportCallback, Value * pipelineHandle, Value * portNum) const  {

    // TODO: if we keep the buffer fd, manual indicates we can resize it by calling ftruncate again. How would this
    // affect users of the buffer prior to setting the new addr/capacity? The mmap'ed buffer pointing to the fds
    // have a set size that may mean its a non-issue as long as ftruncating the buffer doesn't invalidate them.

    Module * const m = b.getModule();
    auto & DL = m->getDataLayout();

    const auto traceDynamicBuffer = (reportCallback != nullptr);

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    assert ("unspecified module" && b.getModule());
    assert (mLinear);

    name << "__FdBackedDynamicBuffer";
    name << "_reserve_capacity" << mAddressSpace;
    if (LLVM_UNLIKELY(traceDynamicBuffer)) {
        name << 'T';
    }

    PointerType * const voidPtrTy = b.getVoidPtrTy();
    const auto blockWidth = b.getBitBlockWidth();

    Function * f = m->getFunction(name.str());

    if (f == nullptr) {

        const auto ip = b.saveIP();

        StructType * const handleTy = getHandleType(b);
        PointerType * const handlePtrTy = handleTy->getPointerTo(mAddressSpace);

        PointerType * const i8PtrTy = b.getInt8PtrTy();
        const auto voidPtrTyAlign = DL.getABITypeAlign(i8PtrTy).value();

        auto & C = b.getContext();
        IntegerType * const intPtrTy = DL.getIntPtrType(C);
        const auto intPtrTyAlign = DL.getABITypeAlign(intPtrTy).value();

        ConstantInt * const BLOCK_WIDTH = b.getSize(blockWidth);

        ConstantInt * const i32_ZERO = b.getInt32(0);

        SmallVector<Type *, 7> paramTypes(traceDynamicBuffer ? 7 : 4);
        paramTypes[0] = voidPtrTy; // shared struct ptr
        paramTypes[1] = intPtrTy; // produced
        paramTypes[2] = intPtrTy; // required
        paramTypes[3] = intPtrTy; // blocksPerChunk
        if (LLVM_UNLIKELY(traceDynamicBuffer)) {
            paramTypes[4] = voidPtrTy; // reportExpansionCallback
            paramTypes[5] = voidPtrTy; // pipelineHandle
            paramTypes[6] = intPtrTy; // portNum
        }

        FunctionType * funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

        f = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            f->setHasUWTable();
            #else
            f->setUWTableKind(UWTableKind::Default);
            #endif
        }

        BasicBlock * const entry = BasicBlock::Create(C, "entry", f);
        BasicBlock * const addPendingDeletion = BasicBlock::Create(C, "addPendingDeletion", f);
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
        Value * const produced = nextArg();
        produced->setName("produced");
        Value * const required = nextArg();
        required->setName("required");
        Value * const bytesPerChunk = nextArg();
        bytesPerChunk->setName("bytesPerChunk");
        Value * reportExpansionCallback = nullptr;
        Value * pipelineHandle = nullptr;
        Value * portNum = nullptr;
        if (LLVM_UNLIKELY(traceDynamicBuffer)) {
            reportExpansionCallback = nextArg();
            reportExpansionCallback->setName("reportExpansionCallback");
            pipelineHandle = nextArg();
            pipelineHandle->setName("pipelineHandle");
            portNum = nextArg();
            portNum->setName("portNum");
        }
        assert (arg == f->arg_end());

        Value * const requiredChunks = b.CreateCeilUDiv(b.CreateAdd(produced, required), BLOCK_WIDTH);

        FixedArray<Value *, 2> indices;
        indices[0] = i32_ZERO;
        indices[1] = b.getInt32(LinearMallocedAddress);
        Value * const addrField = b.CreateInBoundsGEP(handleTy, handle, indices);
        Value * const initialAddr = b.CreateAlignedLoad(i8PtrTy, addrField, voidPtrTyAlign);
        indices[1] = b.getInt32(LinearCapacity);
        Value * const capacityField = b.CreateInBoundsGEP(handleTy, handle, indices);
        Value * const initialCapacity = b.CreateAlignedLoad(intPtrTy, capacityField, intPtrTyAlign);

        indices[1] = b.getInt32(FDDB_Field::Fd);
        Value * const fdField = b.CreateInBoundsGEP(handleTy, handle, indices);
        IntegerType * const intTy = b.getIntNTy(8 * sizeof(int));
        const auto intTyAlign = DL.getABITypeAlign(intTy).value();
        Value * const fd = b.CreateAlignedLoad(intTy, fdField, intTyAlign);
        assert (blockWidth > 8);

        FixedArray<Value *, 4> args;
        args[0] = fd;
        args[1] = initialAddr;
        args[2] = b.CreateMul(initialCapacity, bytesPerChunk);

        // allocate a buffer with at least twice the capacity and copy the unconsumed data back into it
        Value * const expandedCapacity = b.CreateAdd(initialCapacity, b.CreateRoundUp(requiredChunks, initialCapacity));
        Value * const expandedBytes = b.CreateMul(expandedCapacity, bytesPerChunk);

        args[3] = expandedBytes;

        Function * const makeBuffer = m->getFunction(__RESIZE_FD_BACKED_BUFFER); assert (makeBuffer);
        Value * const expandedAddr = b.CreatePointerCast(b.CreateCall(makeBuffer, args), i8PtrTy);

        b.CreateAlignedStore(expandedAddr, addrField, voidPtrTyAlign);
        b.CreateAlignedStore(expandedCapacity, capacityField, intPtrTyAlign);

        b.CreateLikelyCondBr(b.CreateICmpNE(expandedAddr, initialAddr), addPendingDeletion, exit);

        b.SetInsertPoint(addPendingDeletion);

        if (LLVM_UNLIKELY(traceDynamicBuffer)) {

            FixedArray<Type *, 4> paramTypes;
            paramTypes[0] = voidPtrTy; // pipeline handle
            paramTypes[1] = intPtrTy; // port num
            paramTypes[2] = intPtrTy; // produced
            paramTypes[3] = intPtrTy; // capacity

            FunctionType * funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

            FixedArray<Value *, 4> callbackArgs;
            callbackArgs[0] = pipelineHandle;
            callbackArgs[1] = portNum;
            callbackArgs[2] = produced;
            callbackArgs[3] = b.CreateMul(expandedCapacity, BLOCK_WIDTH);

            b.CreateCall(funcTy, b.CreatePointerCast(reportExpansionCallback, funcTy->getPointerTo()), callbackArgs);

        }

        FixedArray<Value *, 2> indices2;
        indices2[0] = i32_ZERO;
        indices2[1] = b.getInt32(PendingDeletionStruct);
        Value * const pendingField = b.CreateGEP(handleTy, handle, indices2);
        Value * const safeToDeleteAt = b.CreateAdd(produced, BLOCK_WIDTH);
        addToPendingDeletions(b, pendingField, initialAddr, initialCapacity, bytesPerChunk, safeToDeleteAt, mAddressSpace);

        b.CreateBr(exit);

        b.SetInsertPoint(exit);
        b.CreateRetVoid();

        b.restoreIP(ip);

    }

    SmallVector<Value *, 7> args(traceDynamicBuffer ? 7 : 4);
    args[0] = b.CreatePointerCast(mHandle, voidPtrTy);
    args[1] = produced;
    args[2] = required;
    args[3] = b.getSize(b.getTypeSize(DL, mType));
    if (LLVM_UNLIKELY(traceDynamicBuffer)) {
        args[4] = reportCallback; // reportExpansionCallback
        args[5] = pipelineHandle; // pipelineHandle
        args[6] = portNum; // portNum
    }
    b.CreateCall(f, args);
    return nullptr;
}

void FdBackedDynamicBuffer::freePendingDeletions(KernelBuilder & b, llvm::Value * consumed) const {
    FixedArray<Value *, 2> indices2;
    indices2[0] = b.getInt32(0);
    indices2[1] = b.getInt32(PendingDeletionStruct);
    Value * const handle = b.CreateGEP(mHandleType, mHandle, indices2);
    removeFromPendingDeletions(b, handle, consumed, false, mAddressSpace);
}

Value * FdBackedDynamicBuffer::getMallocAddress(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling getBaseAddress");
    Value * const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(FDDB_Field::LinearMallocedAddress)});
    Module * const m = b.getModule();
    auto & DL = m->getDataLayout();
    PointerType * const ptrTy = getPointerType();
    const auto ptrTyAlign = DL.getABITypeAlign(ptrTy).value();
    return b.CreateAlignedLoad(ptrTy, p, ptrTyAlign);
}

Value * FdBackedDynamicBuffer::getBaseAddress(KernelBuilder & b) const {
    return getMallocAddress(b);
}

void FdBackedDynamicBuffer::setBaseAddress(KernelBuilder & b, Value * const addr) const {
    unsupported("setBaseAddress", "FdBackedDynamic");
}

Value * FdBackedDynamicBuffer::getInternalCapacity(KernelBuilder & b) const {
    assert (mHandle && "has not been set prior to calling getCapacity");
    Module * const m = b.getModule();
    auto & DL = m->getDataLayout();
    IntegerType * sizeTy = b.getSizeTy();
    const auto sizeTyAlign = DL.getABITypeAlign(sizeTy).value();
    Value * const p = b.CreateInBoundsGEP(mHandleType, mHandle, {b.getInt32(0), b.getInt32(FDDB_Field::LinearCapacity)});
    Value * const cap = b.CreateAlignedLoad(sizeTy, p, sizeTyAlign);
    Constant * const LOG_2_BLOCK_WIDTH = b.getSize(floor_log2(b.getBitBlockWidth()));
    return b.CreateShl(cap, LOG_2_BLOCK_WIDTH);
}

Value * FdBackedDynamicBuffer::getCapacity(KernelBuilder & b) const {
    return getInternalCapacity(b);
}

void FdBackedDynamicBuffer::setCapacity(KernelBuilder & b, Value * const capacity) const {
    unsupported("setCapacity", "FdBackedDynamic");
}

Value * FdBackedDynamicBuffer::modByCapacity(KernelBuilder & /* b */, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    return offset;
}

Value * FdBackedDynamicBuffer::getVirtualBasePtr(KernelBuilder & /* b */, Value * baseAddress, Value * const /* transferredItems */) const {
    return baseAddress;
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

void RepeatingBuffer::allocateBuffer(KernelBuilder & b, Value * const capacityMultiplier, Value * reportCallback, Value * pipelineHandle, Value * portNum) {
    unsupported("allocateBuffer", "Repeating");
}

void RepeatingBuffer::releaseBuffer(KernelBuilder & b) const {
    unsupported("releaseBuffer", "Repeating");
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

Value * RepeatingBuffer::reserveCapacity(KernelBuilder & /* b */, Value * /* produced */, Value * consumed, Value * const required, Value * /* reportCallback */, Value * /* pipelineHandle */, Value * /* portNum */) const  {
    unsupported("linearCopyBack", "Repeating");
}

// Constructors

ExternalBuffer::ExternalBuffer(const unsigned id, KernelBuilder & b, Type * const type,
                               const unsigned AddressSpace)
: StreamSetBuffer(id, BufferKind::ExternalBuffer, b, type,  true, AddressSpace) {

}

ManagedDynamicBuffer::ManagedDynamicBuffer(const unsigned id, KernelBuilder & b,
                                           Type * const type, const bool linear, const unsigned AddressSpace)
: InternalBuffer(id, BufferKind::ManagedDynamicBuffer, b, type, linear, AddressSpace) {

}

FdBackedDynamicBuffer::FdBackedDynamicBuffer(const unsigned id, KernelBuilder & b,
                                            Type * const type, const unsigned AddressSpace)
: InternalBuffer(id, BufferKind::FdBackedDynamicBuffer, b, type, true, AddressSpace) {

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
, mLinear(linear)

{

}

StreamSetBuffer::~StreamSetBuffer() { }

}
