/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <llvm/IR/Type.h>  // for Type
#include <llvm/IR/DerivedTypes.h>  // for Type

namespace IDISA { class IDISA_Builder; }
namespace llvm { class Value; }
namespace llvm { class Constant; }

namespace kernel {

class Kernel;
class PipelineKernel;
class KernelBuilder;

class StreamSetBuffer {
public:

    enum class BufferKind : unsigned {
        ExternalBuffer
        , RepeatingBuffer
        , StaticBuffer
        , DynamicBuffer
        , ManagedDynamicBuffer
    };

    using ScalarRef = std::pair<llvm::Value *, llvm::Type *>;

    BufferKind getBufferKind() const {
        return mBufferKind;
    }

    llvm::Type * getType() const {
        return mType;
    }

    llvm::Type * getBaseType() const {
        return mBaseType;
    }

    unsigned getAddressSpace() const {
        return mAddressSpace;
    }

    __attribute__((const)) llvm::PointerType * getPointerType()  const {
        return getType()->getPointerTo(getAddressSpace());
    }

    bool isLinear() const {
        return mLinear;
    }

    unsigned getId() const {
        return mId;
    }

    unsigned getFieldWidth() const;

    bool isEmptySet() const;

    bool isSingleElementStreamSet() const;

    bool isDynamic() const {
        return (mBufferKind == BufferKind::DynamicBuffer) || (mBufferKind == BufferKind::ManagedDynamicBuffer);
    }

    virtual ~StreamSetBuffer() = 0;

    llvm::Value * getHandle() const {
        return mHandle;
    }

    void setHandle(llvm::Value * const handle) const {
        mHandle = handle;
    }

    void setHandle(ScalarRef handle) const {
        mHandle = handle.first;
        assert (handle.second == mHandleType);
    }

    virtual void allocateBuffer(kernel::KernelBuilder & b, llvm::Value * const capacityMultiplier) = 0;

    virtual void releaseBuffer(kernel::KernelBuilder & b) const = 0;

    virtual void destroyBuffer(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value *capacity) const = 0;

    // The number of items that cam be linearly accessed from a given logical stream position.
    virtual llvm::Value * getLinearlyAccessibleItems(kernel::KernelBuilder & b, llvm::Value * fromPosition, llvm::Value * totalItems) const = 0;

    virtual llvm::Value * getLinearlyWritableItems(kernel::KernelBuilder & b, llvm::Value * fromPosition, llvm::Value * consumedItems) const = 0;

    virtual llvm::StructType * getHandleType(kernel::KernelBuilder & b) const = 0;

    llvm::PointerType * getHandlePointerType(kernel::KernelBuilder & b) const {
        return getHandleType(b)->getPointerTo(getAddressSpace());
    }

    virtual llvm::Value * getStreamBlockPtr(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex) const;

    virtual llvm::Value * getStreamPackPtr(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex, llvm::Value * packIndex) const;

    virtual llvm::Value * getStreamSetCount(kernel::KernelBuilder & b) const;

    virtual llvm::Value * getBaseAddress(kernel::KernelBuilder & b) const = 0;

    virtual llvm::Value * getMallocAddress(kernel::KernelBuilder & b) const = 0;

    virtual void setBaseAddress(kernel::KernelBuilder & b, llvm::Value * addr) const = 0;

    virtual void setCapacity(kernel::KernelBuilder & b, llvm::Value * size) const = 0;

    virtual llvm::Value * getCapacity(kernel::KernelBuilder & b) const = 0;

    virtual llvm::Value * getInternalCapacity(kernel::KernelBuilder & b) const = 0;

    virtual llvm::Value * modByCapacity(kernel::KernelBuilder & b, llvm::Value * const offset) const = 0;

    virtual llvm::Value * getRawItemPointer(kernel::KernelBuilder & b, llvm::Value * streamIndex, llvm::Value * absolutePosition) const;

    virtual llvm::Value * getVirtualBasePtr(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value * const transferredItems) const = 0;

    virtual llvm::Value * requiresExpansion(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const = 0;

    virtual void linearCopyBack(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const = 0;

    virtual llvm::Value * expandBuffer(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const = 0;

    static llvm::Type * resolveType(kernel::KernelBuilder & b, llvm::Type * const streamSetType);

    static void linkFunctions(kernel::KernelBuilder & b); // temporary function

    void assertAccessIsWithinStreamSetMemory(kernel::KernelBuilder & b, llvm::Value * ptr, const size_t size, llvm::Value * const start, llvm::Value * const end) const;

protected:

    StreamSetBuffer(const unsigned id, const BufferKind k, kernel::KernelBuilder & b, llvm::Type * baseType, const bool linear, const unsigned AddressSpace);

private:

    void assertValidStreamIndex(kernel::KernelBuilder & b, llvm::Value * streamIndex) const;

protected:

    const unsigned                  mId;
    const BufferKind                mBufferKind;
    // Each StreamSetBuffer object is local to the Kernel (or pipeline) object at (pre-JIT) "compile time" but
    // by sharing the same handle will refer to the same stream set at (post-JIT) run time.
    mutable llvm::Value *           mHandle;
    llvm::Type * const              mType;
    llvm::Type * const              mBaseType;
    mutable llvm::StructType *      mHandleType;
    const unsigned                  mAddressSpace;
    const bool                      mLinear;
};

class ExternalBuffer final : public StreamSetBuffer {
public:
    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() == BufferKind::ExternalBuffer;
    }

    enum Field { BaseAddress, EffectiveCapacity };

    ExternalBuffer(const unsigned id, kernel::KernelBuilder & b, llvm::Type * const type, const unsigned AddressSpace);

    void allocateBuffer(kernel::KernelBuilder & b, llvm::Value * const capacityMultiplier) override;

    void releaseBuffer(kernel::KernelBuilder & b) const override;

    void destroyBuffer(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value *capacity) const override;

    llvm::Value * getVirtualBasePtr(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value * const transferredItems) const override;

    llvm::Value * getLinearlyAccessibleItems(kernel::KernelBuilder & b, llvm::Value * fromPosition, llvm::Value * totalItems) const override;

    llvm::Value * getLinearlyWritableItems(kernel::KernelBuilder & b, llvm::Value * fromPosition, llvm::Value * consumedItems) const override;

    llvm::StructType * getHandleType(kernel::KernelBuilder & b) const override;

    llvm::Value * getBaseAddress(kernel::KernelBuilder & b) const override;

    llvm::Value * getMallocAddress(kernel::KernelBuilder & b) const override;

    void setCapacity(kernel::KernelBuilder & b, llvm::Value * capacity) const override;

    llvm::Value * getCapacity(kernel::KernelBuilder & b) const override;

    llvm::Value * getInternalCapacity(kernel::KernelBuilder & b) const override;

    llvm::Value * modByCapacity(kernel::KernelBuilder & b, llvm::Value * const offset) const override;

    llvm::Value * requiresExpansion(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void linearCopyBack(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    llvm::Value * expandBuffer(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void setBaseAddress(kernel::KernelBuilder & b, llvm::Value * addr) const override;

private:

    void assertValidBlockIndex(kernel::KernelBuilder & b, llvm::Value * blockIndex) const;

};

class InternalBuffer : public StreamSetBuffer {
public:

    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() != BufferKind::ExternalBuffer;
    }

    llvm::Value * getStreamBlockPtr(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex) const final;

    llvm::Value * getStreamPackPtr(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex, llvm::Value * packIndex) const final;

    llvm::Value * getVirtualBasePtr(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value * const transferredItems) const override;

    llvm::Value * getLinearlyAccessibleItems(kernel::KernelBuilder & b, llvm::Value * fromPosition, llvm::Value * const totalItems) const override;

    llvm::Value * getLinearlyWritableItems(kernel::KernelBuilder & b, llvm::Value * fromPosition, llvm::Value * consumedItems) const override;

protected:

    InternalBuffer(const unsigned id, const BufferKind k, kernel::KernelBuilder & b, llvm::Type * baseType,
                   const bool linear, const unsigned AddressSpace);


};

class StaticBuffer final : public InternalBuffer {
public:
    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() == BufferKind::StaticBuffer;
    }

    StaticBuffer(const unsigned id, kernel::KernelBuilder & b, llvm::Type * const type,
                 const size_t capacity,
                 const bool linear, const unsigned AddressSpace);

    enum Field { BaseAddress, EffectiveCapacity, MallocedAddress, InternalCapacity, PriorAddress };

    void allocateBuffer(kernel::KernelBuilder & b, llvm::Value * const capacityMultiplier) override;

    void releaseBuffer(kernel::KernelBuilder & b) const override;

    void destroyBuffer(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value *capacity) const override;

    llvm::StructType * getHandleType(kernel::KernelBuilder & b) const override;

    llvm::Value * getBaseAddress(kernel::KernelBuilder & b) const override;

    llvm::Value * getMallocAddress(kernel::KernelBuilder & b) const override;

    void setBaseAddress(kernel::KernelBuilder & b, llvm::Value * addr) const override;

    void setCapacity(kernel::KernelBuilder & b, llvm::Value * capacity) const override;

    llvm::Value * getCapacity(kernel::KernelBuilder & b) const override;

    llvm::Value * getInternalCapacity(kernel::KernelBuilder & b) const override;

    llvm::Value * modByCapacity(kernel::KernelBuilder & b, llvm::Value * const offset) const override;

    llvm::Value * requiresExpansion(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void linearCopyBack(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    llvm::Value * expandBuffer(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    size_t getCapacity() const {
        return mCapacity;
    }

private:

    const size_t    mCapacity;

};

class DynamicBuffer final : public InternalBuffer {
protected:

    enum { LinearMallocedAddress = 0,
           LinearInternalCapacity = 1,
           LinearBaseAddress = 2,
           LinearEffectiveCapacity = 3,
           LinearFields = 4,
           // -------------------------------
           CircularAddressSelector = 0,
           CircularBaseAddress = 1,
           CircularSecondaryBaseAddress = 2,
           CircularInternalCapacity = 3,
           CircularSecondaryInternalCapacity = 4,
           CircularFields = 5 };

public:

    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() == BufferKind::DynamicBuffer;
    }

    DynamicBuffer(const unsigned id, kernel::KernelBuilder & b, llvm::Type * type, const size_t initialCapacity,
                  const bool hasUnderflow,
                  const bool linear, const unsigned AddressSpace);

    void allocateBuffer(kernel::KernelBuilder & b, llvm::Value * const capacityMultiplier) override;

    void releaseBuffer(kernel::KernelBuilder & b) const override;

    void destroyBuffer(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value *capacity) const override;

    llvm::Value * getMallocAddress(kernel::KernelBuilder & b) const override;

    llvm::Value * getCapacity(kernel::KernelBuilder & b) const override;

    llvm::Value * getInternalCapacity(kernel::KernelBuilder & b) const override;

    void setCapacity(kernel::KernelBuilder & b, llvm::Value * capacity) const override;

    llvm::Value * modByCapacity(kernel::KernelBuilder & b, llvm::Value * const offset) const override;

    llvm::Value * requiresExpansion(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void linearCopyBack(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    llvm::Value * expandBuffer(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    size_t getInitialCapacity() const {
        return mInitialCapacity;
    }

    llvm::StructType * getHandleType(kernel::KernelBuilder & b) const override;

    llvm::Value * getBaseAddress(kernel::KernelBuilder & b) const override;

    void setBaseAddress(kernel::KernelBuilder & b, llvm::Value * addr) const override;

private:

    const size_t    mInitialCapacity;
    const bool      mHasUnderflow;
};

class ManagedDynamicBuffer final : public InternalBuffer {
public:

    enum { LinearMallocedAddress = 0,
           LinearInternalCapacity = 1,
           LinearBaseAddress = 2,
           LinearEffectiveCapacity = 3,
           LinearFields = 4 };

    enum ThreadLocalField { PriorAddress, PriorCapacity, NewAddress };

    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() == BufferKind::ManagedDynamicBuffer;
    }

    ManagedDynamicBuffer(const unsigned id, kernel::KernelBuilder & b, llvm::Type * const type, const size_t initialCapacity, const unsigned AddressSpace);

    static llvm::StructType * getInternalThreadLocalHandleType(kernel::KernelBuilder & b);

    void freePendingDeletion(kernel::KernelBuilder & b, llvm::Value * threadLocalHandle) const;

    llvm::StructType * getThreadLocalHandleType(kernel::KernelBuilder & b) const {
        return getInternalThreadLocalHandleType(b);
    }

    llvm::StructType * getHandleType(kernel::KernelBuilder & b) const override;

    void allocateBuffer(kernel::KernelBuilder & b, llvm::Value * const capacityMultiplier) override;

    void releaseBuffer(kernel::KernelBuilder & b) const override;

    void destroyBuffer(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value *capacity) const override;

    llvm::Value * getMallocAddress(kernel::KernelBuilder & b) const override;

    llvm::Value * getCapacity(kernel::KernelBuilder & b) const override;

    llvm::Value * getInternalCapacity(kernel::KernelBuilder & b) const override;

    void setCapacity(kernel::KernelBuilder & b, llvm::Value * capacity) const override;

    llvm::Value * modByCapacity(kernel::KernelBuilder & b, llvm::Value * const offset) const override;

    llvm::Value * getVirtualBasePtr(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value * const transferredItems) const override;

    llvm::Value * getBaseAddress(kernel::KernelBuilder & b) const override;

    void setBaseAddress(kernel::KernelBuilder & b, llvm::Value * addr) const override;

    llvm::Value * requiresExpansion(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void linearCopyBack(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;


    llvm::Value * expandBuffer(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    llvm::Value * getThreadLocalHandle() const {
        return mThreadLocalHandle;
    }

    void setThreadLocalHandle(llvm::Value * const handle) const {
        mThreadLocalHandle = handle;
    }

    size_t getInitialCapacity() const {
        return mInitialCapacity;
    }

private:

    const size_t    mInitialCapacity;

    mutable llvm::Value * mThreadLocalHandle;


};

class RepeatingBuffer final : public InternalBuffer {
public:
    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() == BufferKind::RepeatingBuffer;
    }

    enum Field { BaseAddress };

    RepeatingBuffer(const unsigned id, kernel::KernelBuilder & b, llvm::Type * const type, const bool unaligned);

    llvm::Value * modByCapacity(kernel::KernelBuilder & b, llvm::Value * const offset) const override;

    llvm::Value * getVirtualBasePtr(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value * const transferredItems) const override;

    void allocateBuffer(kernel::KernelBuilder & b, llvm::Value * const capacityMultiplier) override;

    void releaseBuffer(kernel::KernelBuilder & b) const override;

    void destroyBuffer(kernel::KernelBuilder & b, llvm::Value * baseAddress, llvm::Value *capacity) const override;

    llvm::StructType * getHandleType(kernel::KernelBuilder & b) const override;

    llvm::Value * getBaseAddress(kernel::KernelBuilder & b) const override;

    llvm::Value * getMallocAddress(kernel::KernelBuilder & b) const override;

    void setBaseAddress(kernel::KernelBuilder & b, llvm::Value * addr) const override;

    void setCapacity(kernel::KernelBuilder & b, llvm::Value * capacity) const override;

    llvm::Value * getCapacity(kernel::KernelBuilder & b) const override;

    llvm::Value * getInternalCapacity(kernel::KernelBuilder & b) const override;

    llvm::Value * requiresExpansion(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void linearCopyBack(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    llvm::Value * expandBuffer(kernel::KernelBuilder & b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void setModulus(llvm::Value * const modulus) {
        mModulus = modulus;
    }

    llvm::Value * getModulus() const {
        return mModulus;
    }

private:

    llvm::Value * mModulus;
    const bool mUnaligned;

};

}
