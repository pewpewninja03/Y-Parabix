#pragma once

#include <kernel/core/kernel.h>
#include <kernel/core/binding_map.hpp>
#include <kernel/core/kernel_builder.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/Compiler.h>
#include <kernel/illustrator/illustrator_binding.h>

namespace kernel {

class KernelCompiler {

    friend class PipelineCompiler;
    friend class KernelBuilder;
    friend class Kernel;

public:

    using Rational = ProcessingRate::Rational;

    using ScalarRef = std::pair<llvm::Value *, llvm::Type *>;

    using ScalarValueMap = llvm::StringMap<ScalarRef>;

    using ScalarAliasMap = std::vector<std::pair<std::string, std::string>>;

    using ScalarType = Kernel::ScalarType;

    using InternalScalar = Kernel::InternalScalar;

    using InternalScalars = Kernel::InternalScalars;

    using PortType = Kernel::PortType;

    using StreamSetPort = Kernel::StreamSetPort;

    using InitArgs = Kernel::InitArgs;

    using InitArgTypes = Kernel::InitArgTypes;

    using ParamMap = Kernel::ParamMap;

    using ArgIterator = llvm::Function::arg_iterator;

    using MemoryOrdering = KernelBuilder::MemoryOrdering;

    template <typename T, unsigned n = 16>
    using Vec = llvm::SmallVector<T, n>;

    using OwnedStreamSetBuffers = Vec<std::unique_ptr<StreamSetBuffer>>;

    enum class InitializeOptions {
        DoNotIncludeThreadLocalScalars
        , IncludeThreadLocalScalars
        , IncludeAndAutomaticallyAccumulateThreadLocalScalars
    };

public:

    // constructor
    KernelCompiler(not_null<Kernel *> kernel) noexcept;

    void generateKernel(KernelBuilder & b);

    virtual ~KernelCompiler();

public:

    llvm::Value * getHandle() const {
        return mSharedHandle;
    }

    void setHandle(llvm::Value * const handle) {
        mSharedHandle = handle;
    }

    llvm::Value * getThreadLocalHandle() const {
        return mThreadLocalHandle;
    }

    void setThreadLocalHandle(llvm::Value * const handle) {
        mThreadLocalHandle = handle;
    }

public:

    LLVM_READNONE const std::string & getName() const {
        return mTarget->getName();
    }

    LLVM_READNONE llvm::Value * getAccessibleInputItems(const llvm::StringRef name) const {
        return getAccessibleInputItems(getBinding(BindingType::StreamInput, name).Index);
    }

    LLVM_READNONE llvm::Value * getAccessibleInputItems(const unsigned index) const {
        assert (index < mAccessibleInputItems.size());
        return mAccessibleInputItems[index];
    }

    LLVM_READNONE llvm::Value * getAvailableInputItems(const llvm::StringRef name) const {
        return getAvailableInputItems(getBinding(BindingType::StreamInput, name).Index);
    }

    LLVM_READNONE llvm::Value * getAvailableInputItems(const unsigned index) const {
        assert (index < mAvailableInputItems.size());
        return mAvailableInputItems[index];
    }

    LLVM_READNONE bool canSetTerminateSignal() const {
        return mTarget->canSetTerminateSignal();
    }

    LLVM_READNONE llvm::Value * getTerminationSignalPtr() const {
        return mTerminationSignalPtr;
    }

    LLVM_READNONE llvm::Value * getProcessedInputItemsPtr(const llvm::StringRef name) const {
        return getProcessedInputItemsPtr(getBinding(BindingType::StreamInput, name).Index);
    }

    LLVM_READNONE llvm::Value * getProcessedInputItemsPtr(const unsigned index) const {
        return mProcessedInputItemPtr[index];
    }

    LLVM_READNONE llvm::Value * getProducedOutputItemsPtr(const llvm::StringRef name) const {
        return getProducedOutputItemsPtr(getBinding(BindingType::StreamOutput, name).Index);
    }

    LLVM_READNONE llvm::Value * getProducedOutputItemsPtr(const unsigned index) const {
        return mProducedOutputItemPtr[index];
    }

    LLVM_READNONE llvm::Value * getWritableOutputItems(const llvm::StringRef name) const {
        return getWritableOutputItems(getBinding(BindingType::StreamOutput, name).Index);
    }

    LLVM_READNONE llvm::Value * getWritableOutputItems(const unsigned index) const {
        return mWritableOutputItems[index];
    }

    LLVM_READNONE llvm::Value * getConsumedOutputItems(const llvm::StringRef name) const {
        return getConsumedOutputItems(getBinding(BindingType::StreamOutput, name).Index);
    }

    LLVM_READNONE llvm::Value * getConsumedOutputItems(const unsigned index) const {
        return mConsumedOutputItems[index];
    }

    llvm::Value * getNumOfStrides() const {
        return mNumOfStrides;
    }

    void setTerminationSignalPtr(llvm::Value * ptr) {
        mTerminationSignalPtr = ptr;
    }

    llvm::Value * getExternalSegNo() const {
        return mExternalSegNo;
    }

    void setExternalSegNo(llvm::Value * segNo) {
        mExternalSegNo = segNo;
    }

    LLVM_READNONE llvm::Value * isFinal() const {
        return mIsFinal;
    }

    // Binding + Relationship get/set functions

    LLVM_READNONE const Bindings & getInputStreamSetBindings() const {
        return mInputStreamSets;
    }

    LLVM_READNONE const Binding & getInputStreamSetBinding(const unsigned i) const {
        assert (i < getNumOfStreamInputs());
        return mInputStreamSets[i];
    }

    LLVM_READNONE const Binding & getInputStreamSetBinding(const llvm::StringRef name) const {
        return mTarget->getInputStreamSetBinding(getBinding(BindingType::StreamInput, name).Index);
    }

    LLVM_READNONE StreamSet * getInputStreamSet(const unsigned i) const {
        return static_cast<StreamSet *>(getInputStreamSetBinding(i).getRelationship());
    }

    LLVM_READNONE StreamSet * getInputStreamSet(const llvm::StringRef name) const {
        return static_cast<StreamSet *>(getInputStreamSetBinding(name).getRelationship());
    }

    void setInputStreamSet(const llvm::StringRef name, StreamSet * value) const {
        mTarget->setInputStreamSetAt(getBinding(BindingType::StreamInput, name).Index, value);
    }

    LLVM_READNONE size_t getNumOfStreamInputs() const {
        return mInputStreamSets.size();
    }

    LLVM_READNONE StreamSetBuffer * getInputStreamSetBuffer(const unsigned i) const {
        assert (i < mStreamSetInputBuffers.size());
        assert (mStreamSetInputBuffers[i]);
        return mStreamSetInputBuffers[i].get();
    }

    LLVM_READNONE StreamSetBuffer * getInputStreamSetBuffer(const llvm::StringRef name) const {
        return getInputStreamSetBuffer(getBinding(BindingType::StreamInput, name).Index);
    }

    LLVM_READNONE const Bindings & getOutputStreamSetBindings() const {
        return mOutputStreamSets;
    }

    LLVM_READNONE const Binding & getOutputStreamSetBinding(const unsigned i) const {
        assert (i < getNumOfStreamOutputs());
        return mOutputStreamSets[i];
    }

    LLVM_READNONE const Binding & getOutputStreamSetBinding(const llvm::StringRef name) const {
        return mTarget->getOutputStreamSetBinding(getBinding(BindingType::StreamOutput, name).Index);
    }

    LLVM_READNONE StreamSet * getOutputStreamSet(const unsigned i) const {
        return llvm::cast<StreamSet>(getOutputStreamSetBinding(i).getRelationship());
    }

    LLVM_READNONE StreamSet * getOutputStreamSet(const llvm::StringRef name) const {
        return llvm::cast<StreamSet>(getOutputStreamSetBinding(name).getRelationship());
    }

    void setOutputStreamSet(const llvm::StringRef name, StreamSet * value) const {
        mTarget->setOutputStreamSetAt(getBinding(BindingType::StreamOutput, name).Index, value);
    }

    LLVM_READNONE size_t getNumOfStreamOutputs() const {
        return mOutputStreamSets.size();
    }

    LLVM_READNONE StreamSetBuffer * getOutputStreamSetBuffer(const unsigned i) const {
        assert (i < mStreamSetOutputBuffers.size());
        assert (mStreamSetOutputBuffers[i]);
        return mStreamSetOutputBuffers[i].get();
    }

    LLVM_READNONE StreamSetBuffer * getOutputStreamSetBuffer(const llvm::StringRef name) const {
        return getOutputStreamSetBuffer(getBinding(BindingType::StreamOutput, name).Index);
    }

    LLVM_READNONE const Bindings & getInputScalarBindings() const {
        return mInputScalars;
    }

    LLVM_READNONE const Binding & getInputScalarBinding(const unsigned i) const {
        assert (i < mInputScalars.size());
        return mInputScalars[i];
    }

    LLVM_READNONE const Binding & getInputScalarBinding(const llvm::StringRef name) const {
        return mTarget->getInputScalarBinding(getBinding(BindingType::ScalarInput, name).Index);
    }

    LLVM_READNONE Scalar * getInputScalar(const unsigned i) const {
        return llvm::cast<Scalar>(getInputScalarBinding(i).getRelationship());
    }

    LLVM_READNONE Scalar * getInputScalar(const llvm::StringRef name) const {
        return llvm::cast<Scalar>(getInputScalarBinding(name).getRelationship());
    }

    LLVM_READNONE size_t getNumOfScalarInputs() const {
        return mInputScalars.size();
    }

    LLVM_READNONE const Bindings & getOutputScalarBindings() const {
        return mOutputScalars;
    }

    LLVM_READNONE const Binding & getOutputScalarBinding(const unsigned i) const {
        assert (i < mOutputScalars.size());
        return mOutputScalars[i];
    }

    LLVM_READNONE const Binding & getOutputScalarBinding(const llvm::StringRef name) const {
        return mTarget->getOutputScalarBinding(getBinding(BindingType::ScalarOutput, name).Index);
    }

    LLVM_READNONE size_t getNumOfScalarOutputs() const {
        return mOutputScalars.size();
    }

    LLVM_READNONE Scalar * getOutputScalar(const unsigned i) const {
        return llvm::cast<Scalar>(getOutputScalarBinding(i).getRelationship());
    }

    LLVM_READNONE Scalar * getOutputScalar(const llvm::StringRef name) const {
        return llvm::cast<Scalar>(getOutputScalarBinding(name).getRelationship());
    }

    // StreamSet/Scalar mapping functions

    LLVM_READNONE StreamSetPort getStreamPort(const llvm::StringRef name) const;

    LLVM_READNONE const StreamSetBuffer * getStreamSetBuffer(const llvm::StringRef name) const {
        const auto port = getStreamPort(name);
        if (port.Type == PortType::Input) {
            return getInputStreamSetBuffer(port.Number);
        } else {
            return getOutputStreamSetBuffer(port.Number);
        }
    }

    LLVM_READNONE const Binding & getStreamBinding(const llvm::StringRef name) const {
        return getStreamBinding(getStreamPort(name));
    }

    LLVM_READNONE const Binding & getStreamBinding(const StreamSetPort port) const {
        return (port.Type == PortType::Input) ? getInputStreamSetBinding(port.Number) : getOutputStreamSetBinding(port.Number);
    }

    LLVM_READNONE ProcessingRate::Rational getLowerBound(const Binding & binding) const;

    LLVM_READNONE ProcessingRate::Rational getUpperBound(const Binding & binding) const;

    LLVM_READNONE bool requiresOverflow(const Binding & binding) const;

    bool hasScalarField(const llvm::StringRef name) const;

    ScalarRef getScalarFieldPtr(KernelBuilder & b, const llvm::StringRef name) const;

    LLVM_READNONE const BindingMapEntry & getBinding(const BindingType type, const llvm::StringRef name) const;

protected:

    virtual void constructStreamSetBuffers(KernelBuilder & b);

    virtual void addBaseInternalProperties(KernelBuilder & b);

    ScalarRef getThreadLocalScalarFieldPtr(KernelBuilder & b, llvm::Value * handle, const llvm::StringRef name) const;

private:

    void initializeScalarMap(KernelBuilder & b, const InitializeOptions options);

    void initializeIOBindingMap();

    void initializeOwnedBufferHandles(KernelBuilder & b, const InitializeOptions options, llvm::Value * expectedNumOfStrides = nullptr);

protected:

    void addAlias(llvm::StringRef alias, llvm::StringRef scalarName);

public:

    void callGenerateInitializeMethod(KernelBuilder & b);

    virtual void callGenerateExpectedOutputSizeMethod(KernelBuilder & b);

    virtual void bindAdditionalInitializationArguments(KernelBuilder & b, ArgIterator & arg, const ArgIterator & arg_end);

    void callGenerateInitializeThreadLocalMethod(KernelBuilder & b);

    void callGenerateAllocateSharedInternalStreamSets(KernelBuilder & b);

    void callGenerateAllocateThreadLocalInternalStreamSets(KernelBuilder & b);

    std::vector<llvm::Value *> getDoSegmentProperties(KernelBuilder & b) const;

    void setDoSegmentProperties(KernelBuilder & b, const llvm::ArrayRef<llvm::Value *> args);

    void callGenerateDoSegmentMethod(KernelBuilder & b);

    void callGenerateFinalizeThreadLocalMethod(KernelBuilder & b);

    void callGenerateFinalizeMethod(KernelBuilder & b);

    static Rational getLCMOfFixedRateInputs(const Kernel * const target);

protected:

    void registerIllustrator(KernelBuilder & b, llvm::Constant * kernelName, llvm::Constant * streamName, const size_t rows, const size_t cols, const size_t itemWidth, const MemoryOrdering ordering, IllustratorTypeId illustratorTypeId, const char replacement0, const char replacement1, const llvm::ArrayRef<size_t> loopIds) const;

    void registerIllustrator(KernelBuilder & b, llvm::Value * illustratorObject, llvm::Constant * kernelName, llvm::Constant * streamName, llvm::Value * handle, const size_t rows, const size_t cols, const size_t itemWidth, const MemoryOrdering ordering, IllustratorTypeId illustratorTypeId, const char replacement0, const char replacement1, const llvm::ArrayRef<size_t> loopIds) const;

    void captureStreamData(KernelBuilder & b, llvm::Constant * kernelName, llvm::Constant * streamName, llvm::Value * handle, llvm::Value * strideNum, llvm::Type * type, const MemoryOrdering ordering, llvm::Value * streamData, llvm::Value * from, llvm::Value * to) const;

protected:

    llvm::Value * getReportExpansionCallback() const {
        return mReportExpansionCallback;
    }

    llvm::Value * getPipelineHandle() const {
        return mPipelineHandle;
    }

protected:

    virtual std::vector<llvm::Value *> getFinalOutputScalars(KernelBuilder & b);

private:

    void clearInternalStateAfterCodeGen();

    void runAllOptimizationPasses(KernelBuilder & b, Kernel::SelectedOptimizationPasses & passes);

protected:

    Kernel * const                  mTarget;

    const Bindings &                mInputStreamSets;
    const Bindings &                mOutputStreamSets;

    const Bindings &                mInputScalars;
    const Bindings &                mOutputScalars;
    const InternalScalars &         mInternalScalars;

    llvm::Function *                mCurrentMethod = nullptr;

    llvm::BasicBlock *              mEntryPoint = nullptr;

    llvm::Value *                   mSharedHandle = nullptr;
    llvm::Value *                   mThreadLocalHandle = nullptr;
    // When finalizing thread-local data, the user may want to summarize it.
    // The common thread local handle points to the first thread's
    // thread-local handle.
    llvm::Value *                   mCommonThreadLocalHandle = nullptr;

    llvm::Value *                   mTerminationSignalPtr = nullptr;
    llvm::Value *                   mIsFinal = nullptr;
    llvm::Value *                   mRawNumOfStrides = nullptr;
    llvm::Value *                   mNumOfStrides = nullptr;
    llvm::Value *                   mFixedRateFactor = nullptr;
    llvm::Value *                   mExternalSegNo = nullptr;
    #ifdef ENABLE_PAPI
    llvm::Value *                   mPAPIEventSetId = nullptr;
    #endif

    llvm::Value *                   mReportExpansionCallback = nullptr;
    llvm::Value *                   mPipelineHandle = nullptr;

    Vec<llvm::Value *>              mInputIsClosed;

    Vec<llvm::Value *>              mProcessedInputItemPtr;

    Vec<llvm::Value *>              mAccessibleInputItems;
    Vec<llvm::Value *>              mAvailableInputItems;
    Vec<llvm::Value *>              mProducedOutputItemPtr;
    Vec<llvm::Value *>              mUpdatableOutputBaseVirtualAddressPtr;
    Vec<llvm::Value *>              mUpdatableOutputCapacityPtr;
    Vec<llvm::Value *>              mInitiallyProducedOutputItems;

    Vec<llvm::Value *>              mWritableOutputItems;
    Vec<llvm::Value *>              mConsumedOutputItems;

    ScalarValueMap                  mScalarFieldMap;
    ScalarAliasMap                  mScalarAliasMap;
    BindingMap                      mBindingMap;

    OwnedStreamSetBuffers           mStreamSetInputBuffers;
    OwnedStreamSetBuffers           mStreamSetOutputBuffers;

};

}

