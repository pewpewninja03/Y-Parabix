#pragma once

#include <kernel/pipeline/pipeline_kernel.h>
#include <llvm/IR/Constants.h>
#include <boost/integer.hpp>

class BaseDriver;

namespace kernel {

class OptimizationBranchBuilder;

class PipelineBuilder : public LLVMTypeSystemInterface {
    friend class PipelineKernel;
    friend class PipelineAnalysis;
    friend class PipelineCompiler;
    friend class OptimizationBranchBuilder;
public:

    using Kernels = PipelineKernel::Kernels;
    using Relationships = PipelineKernel::Relationships;
    using CallBinding = PipelineKernel::CallBinding;
    using CallBindings = PipelineKernel::CallBindings;
    using LengthAssertion = PipelineKernel::LengthAssertion;
    using LengthAssertions = PipelineKernel::LengthAssertions;

    BaseDriver & getDriver() { return mDriver;}

    template<typename KernelType, typename... Args>
    Kernel * CreateKernelCall(Args &&... args) {
        return initializeKernel(new KernelType(mDriver, std::forward<Args>(args) ...), PipelineKernel::KernelBindingFlag::None);
    }

    template<typename KernelType, typename... Args>
    Kernel * CreateKernelFamilyCall(Args &&... args) {
        return initializeKernel(new KernelType(mDriver, std::forward<Args>(args) ...), PipelineKernel::KernelBindingFlag::Family);
    }

    Kernel * AddKernelCall(Kernel * kernel) {
        return initializeKernel(kernel, PipelineKernel::KernelBindingFlag::None);
    }

    Kernel * AddKernelFamilyCall(Kernel * kernel) {
        return initializeKernel(kernel, PipelineKernel::KernelBindingFlag::Family);
    }

    std::shared_ptr<OptimizationBranchBuilder>
        CreateOptimizationBranch(Relationship * const condition,
                                 Bindings && stream_inputs = {}, Bindings && stream_outputs = {},
                                 Bindings && scalar_inputs = {}, Bindings && scalar_outputs = {});

    StreamSet * CreateStreamSet(const unsigned NumElements = 1, const unsigned FieldWidth = 1) {
        return mDriver.CreateStreamSet(NumElements, FieldWidth);
    }

    Scalar * CreateConstant(llvm::Constant * value) {
        return mDriver.CreateConstant(value);
    }

    Scalar * CreateScalar(llvm::Type * type) {
        return mDriver.CreateScalar(type);
    }

    using pattern_t = std::vector<uint64_t>;

    #define RETURN_REPSTREAMSET(...) \
        RepeatingStreamSet * const ss = mDriver.CreateRepeatingStreamSet(__VA_ARGS__); \
        mTarget->mInternallyGeneratedStreamSets.push_back(ss); \
        return ss

    RepeatingStreamSet * CreateRepeatingStreamSet(unsigned FieldWidth, pattern_t string, const bool isDynamic = true) {
        RETURN_REPSTREAMSET(FieldWidth, std::vector<pattern_t>{std::move(string)}, isDynamic);
    }

    RepeatingStreamSet * CreateRepeatingStreamSet(unsigned FieldWidth, std::vector<pattern_t> string, const bool isDynamic = true) {
        RETURN_REPSTREAMSET(FieldWidth, std::move(string), isDynamic);
    }

    StreamSet * CreateRepeatingBixNum(unsigned bixNumBits, pattern_t nums, const bool isDynamic = true);

    template<unsigned FieldWidth, unsigned NumOfElements>
    RepeatingStreamSet * CreateRepeatingStreamSet(std::array<pattern_t, NumOfElements> & string) {
        RETURN_REPSTREAMSET(FieldWidth, std::vector<pattern_t>{string.begin(), string.end()}, true);
    }

    #undef RETURN_REPSTREAMSET

    #define RETURN_REPSTREAMSET(...) \
        RepeatingStreamSet * const ss = mDriver.CreateUnalignedRepeatingStreamSet(__VA_ARGS__); \
        mTarget->mInternallyGeneratedStreamSets.push_back(ss); \
        return ss

    RepeatingStreamSet * CreateUnalignedRepeatingStreamSet(unsigned FieldWidth, pattern_t string, const bool isDynamic = true) {
        RETURN_REPSTREAMSET(FieldWidth, std::vector<pattern_t>{std::move(string)}, isDynamic);
    }

    RepeatingStreamSet * CreateUnalignedRepeatingStreamSet(unsigned FieldWidth, std::vector<pattern_t> string, const bool isDynamic = true) {
        RETURN_REPSTREAMSET(FieldWidth, std::move(string), isDynamic);
    }

    template<unsigned FieldWidth, unsigned NumOfElements>
    RepeatingStreamSet * CreateUnalignedRepeatingStreamSet(std::array<pattern_t, NumOfElements> & string) {
        RETURN_REPSTREAMSET(FieldWidth, std::vector<pattern_t>{string.begin(), string.end()}, true);
    }

    #undef RETURN_REPSTREAMSET

    TruncatedStreamSet * CreateTruncatedStreamSet(const StreamSet * data) {
        return mDriver.CreateTruncatedStreamSet(data);
    }

    template <typename ExternalFunctionType>
    void CreateCall(std::string name, ExternalFunctionType & functionPtr, std::initializer_list<Scalar *> args) {
        llvm::FunctionType * const type = FunctionTypeBuilder<ExternalFunctionType>::get(mDriver.getContext());
        assert ("FunctionTypeBuilder did not resolve a function type." && type);
        assert ("Function was not provided the correct number of args" && type->getNumParams() == args.size());
        // Since the pipeline kernel module has not been made yet, just record the function info and its arguments.
        mTarget->mCallBindings.emplace_back(std::move(name), type, reinterpret_cast<void *>(&functionPtr), std::move(args));
    }

    StreamSet * getInputStreamSet(const unsigned i) {
        return static_cast<StreamSet *>(mTarget->mInputStreamSets[i].getRelationship());
    }

    StreamSet * getInputStreamSet(const llvm::StringRef name);

    unsigned getNumOfStreamInputs() const {
        assert (mTarget);
        return mTarget->getNumOfStreamInputs();
    }

    StreamSet * getOutputStreamSet(const unsigned i) {
        return static_cast<StreamSet *>(mTarget->mOutputStreamSets[i].getRelationship());
    }

    StreamSet * getOutputStreamSet(const llvm::StringRef name);

    unsigned getNumOfStreamOutputs() const {
        assert (mTarget);
        return mTarget->getNumOfStreamOutputs();
    }

    Scalar * getInputScalar(const unsigned i) {
        return static_cast<Scalar *>(mTarget->mInputScalars[i].getRelationship());
    }

    Scalar * getInputScalar(const llvm::StringRef name);

    Scalar * getOutputScalar(const unsigned i) {
        return static_cast<Scalar *>(mTarget->mOutputScalars[i].getRelationship());
    }

    Scalar * getOutputScalar(const llvm::StringRef name);

    void setOutputScalar(const llvm::StringRef name, Scalar * value);

    void AssertEqualLength(const StreamSet * A, const StreamSet * B) {
        mTarget->mLengthAssertions.emplace_back(LengthAssertion{{A, B}});
    }

    virtual ~PipelineBuilder() {}

    virtual Kernel * makeKernel();

    void setExternallySynchronized(const bool value = true) {
        mExternallySynchronized = value;
    }

    void setUniqueName(std::string name) {
        mTarget->mSignature.swap(name);
    }

    void captureByteData(llvm::StringRef streamName, StreamSet * byteData, char nonASCIIsubstitute = '.');

    void captureBitstream(llvm::StringRef streamName, StreamSet * bitstream, char zeroCh = '.', char oneCh = '1');

    void captureBixNum(llvm::StringRef streamName, StreamSet * bixnum, char hexBase = 'A');

    llvm::LLVMContext & getContext() const final {
        return mDriver.getContext();
    }

    unsigned getBitBlockWidth() const final {
        return mDriver.getBitBlockWidth();
    }

    llvm::VectorType * getBitBlockType() const final {
        return mDriver.getBitBlockType();
    }

    llvm::VectorType * getStreamTy(const unsigned FieldWidth = 1) final {
        return mDriver.getStreamTy(FieldWidth);
    }

    llvm::ArrayType * getStreamSetTy(const unsigned NumElements = 1, const unsigned FieldWidth = 1) final {
        return mDriver.getStreamSetTy(NumElements, FieldWidth);
    }

protected:

    PipelineBuilder(BaseDriver & driver, PipelineKernel * const kernel);

    Kernel * initializeKernel(Kernel * const kernel, const unsigned flags);

    llvm::Function * addLinkFunction(llvm::Module * mod, llvm::StringRef name, llvm::FunctionType * type, void * functionPtr) const {
        return mDriver.addLinkFunction(mod, name, type, functionPtr);
    }

    bool hasExternalFunction(const llvm::StringRef functionName) const {
        return mDriver.hasExternalFunction(functionName);
    }

protected:

    BaseDriver &            mDriver;
    // eventual pipeline configuration
    PipelineKernel * const  mTarget;

    bool                    mExternallySynchronized = false;
};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief PipelineBranchBuilder
 ** ------------------------------------------------------------------------------------------------------------- */
class OptimizationBranchBuilder final : public PipelineBuilder {
    friend class PipelineKernel;
    friend class PipelineBuilder;
public:

    const std::unique_ptr<PipelineBuilder> & getNonZeroBranch() const {
        return mNonZeroBranch;
    }

    const std::unique_ptr<PipelineBuilder> & getAllZeroBranch() const {
        return mAllZeroBranch;
    }

    ~OptimizationBranchBuilder();

protected:

    OptimizationBranchBuilder(BaseDriver & driver, Relationship * const condition,
                              PipelineKernel * const allZero,
                              PipelineKernel * const nonZero,
                              PipelineKernel * const branch);

    Kernel * makeKernel() override;

private:
    Relationship * const             mCondition;
    std::unique_ptr<PipelineBuilder> mNonZeroBranch;
    std::unique_ptr<PipelineBuilder> mAllZeroBranch;
};

}

