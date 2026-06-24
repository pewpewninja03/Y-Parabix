#pragma once

#include <kernel/core/kernel_compiler.h>
#include <kernel/pipeline/optimizationbranch.h>
#include <kernel/pipeline/pipeline_kernel.h>
#include <kernel/core/streamset.h>
#include <kernel/core/kernel_builder.h>
#include <boost/container/flat_map.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <llvm/Support/raw_ostream.h>
#include <llvm/ADT/SmallVector.h>
#include <kernel/core/refwrapper.h>
#include  <array>

using namespace llvm;
using namespace boost;
using namespace boost::container;

namespace kernel {

using BindingRef = RefWrapper<Binding>;

using StreamSetGraph = adjacency_list<vecS, vecS, bidirectionalS, no_property, unsigned>;

struct RelationshipRef {
    unsigned Index;
    StringRef Name;
    BindingRef Binding;
    RelationshipRef() = default;
    RelationshipRef(const unsigned index, StringRef name, const kernel::Binding & binding) : Index(index), Name(name), Binding(binding) { }
};

const static std::string CONTROL_CODE = "C";
const static std::string SHARED_PREFIX = "S";
const static std::string THREAD_LOCAL_PREFIX = "T";

const static std::string EXTERNAL_SEGMENT_NUMBER = "E";

const static std::string ALL_ZERO_PATH_TAKEN_COUNT = "AI";

const static std::string ALL_ZERO_INTERNAL_INFLIGHT_COUNT = "AC";
const static std::string NON_ZERO_INTERNAL_INFLIGHT_COUNT = "NC";

using RelationshipGraph = adjacency_list<vecS, vecS, bidirectionalS, no_property, RelationshipRef>;

using RelationshipCache = flat_map<RelationshipGraph::vertex_descriptor, Value *>;


using PortType = Kernel::PortType;
using StreamSetPort = Kernel::StreamSetPort;

using AttrId = Attribute::KindId;

enum : unsigned {
    BRANCH_INPUT = 0
    , ALL_ZERO_BRANCH = 1
    , NON_ZERO_BRANCH = 2
    , BRANCH_OUTPUT = 3
    , CONDITION_VARIABLE = 4
// ----------------------
    , INITIAL_GRAPH_SIZE = 5
};

#define OTHER_BRANCH(TYPE) (TYPE ^ (ALL_ZERO_BRANCH | NON_ZERO_BRANCH))

static_assert (ALL_ZERO_BRANCH < NON_ZERO_BRANCH, "invalid branch type ordering");
static_assert (OTHER_BRANCH(ALL_ZERO_BRANCH) == NON_ZERO_BRANCH, "invalid branch type ordering");
static_assert (OTHER_BRANCH(NON_ZERO_BRANCH) == ALL_ZERO_BRANCH, "invalid branch type ordering");

using CompilerArray = std::array<std::unique_ptr<KernelCompiler>, 4>;

class OptimizationBranchCompiler final : public KernelCompiler {

    enum class RelationshipType : unsigned {
        StreamSet
        , Scalar
    };

public:

    OptimizationBranchCompiler(KernelBuilder & b, OptimizationBranch * const branch) noexcept;

    void addBranchProperties(KernelBuilder & b);
    void constructStreamSetBuffers(KernelBuilder & b) override;
    void generateInitializeMethod(KernelBuilder & b);
    void generateAllocateSharedInternalStreamSetsMethod(KernelBuilder & b, Value * const expectedNumOfStrides);
    void generateInitializeThreadLocalMethod(KernelBuilder & b);
    void generateAllocateThreadLocalInternalStreamSetsMethod(KernelBuilder & b, Value * const expectedNumOfStrides);
    void generateKernelMethod(KernelBuilder & b);
    void generateFinalizeThreadLocalMethod(KernelBuilder & b);
    void generateFinalizeMethod(KernelBuilder & b);

    std::vector<Value *> getFinalOutputScalars(KernelBuilder & b) override;

private:

    Value * loadSharedHandle(KernelBuilder & b, const unsigned branchType) const;

    Value * loadThreadLocalHandle(KernelBuilder & b, const unsigned branchType) const;

    void allocateOwnedBranchBuffers(KernelBuilder & b, Value * const expectedNumOfStrides, const bool nonLocal);

    Value * getInputScalar(KernelBuilder & b, const unsigned scalar);

    inline unsigned getNumOfInputBindings(const Kernel * const kernel, const RelationshipType type) const;

    const Binding & getInputBinding(const Kernel * const kernel, const RelationshipType type, const unsigned i) const;

    unsigned getNumOfOutputBindings(const Kernel * const kernel, const RelationshipType type) const;

    const Binding & getOutputBinding(const Kernel * const kernel, const RelationshipType type, const unsigned i) const;

    void executeBranch(KernelBuilder & b, const unsigned branchType);

    Value * enterBranch(KernelBuilder & b, const unsigned branchType) const;

    void exitBranch(KernelBuilder & b, const unsigned branchType) const;

    Value * calculateAccessibleOrWritableItems(KernelBuilder & b, const Kernel * const kernel, const Binding & binding, Value * const first, Value * const last, Value * const defaultValue) const;

    Value * calculateFinalOutputItemCounts(KernelBuilder & b, Value * const isFinal, const unsigned branchType);

    RelationshipGraph makeRelationshipGraph(const RelationshipType type) const;

    #ifdef PRINT_DEBUG_MESSAGES
    template <typename ... Args>
    void debugPrint(KernelBuilder & b, Twine format, Args ...args) const;
    #endif

private:

    const Relationship * const          mCondition;

    const std::array<const Kernel *, 4> mBranches;

    Value *                             mFirstBranchPath = nullptr;
    PHINode *                           mBranchDemarcationArray = nullptr;
    Value *                             mBranchDemarcationCount = nullptr;

    PHINode *                           mTerminatedPhi = nullptr;

    #ifdef PRINT_DEBUG_MESSAGES
    Value *                             mThreadId = nullptr;
    #endif

    const RelationshipGraph             mStreamSetGraph;
    const RelationshipGraph             mScalarGraph;
    RelationshipCache                   mScalarCache;

};

template <typename Graph>
LLVM_READNONE
inline typename graph_traits<Graph>::edge_descriptor in_edge(const typename graph_traits<Graph>::vertex_descriptor u, const Graph & G) {
    assert (in_degree(u, G) == 1);
    return *in_edges(u, G).first;
}

template <typename Graph>
LLVM_READNONE
inline typename graph_traits<Graph>::edge_descriptor parent(const typename graph_traits<Graph>::edge_descriptor & e, const Graph & G) {
    return in_edge(source(e, G), G);
}

template <typename Graph>
LLVM_READNONE
inline typename graph_traits<Graph>::edge_descriptor out_edge(const typename graph_traits<Graph>::vertex_descriptor u, const Graph & G) {
    assert (out_degree(u, G) == 1);
    return *out_edges(u, G).first;
}

template <typename Graph>
LLVM_READNONE
inline typename graph_traits<Graph>::edge_descriptor child(const typename graph_traits<Graph>::edge_descriptor & e, const Graph & G) {
    return out_edge(target(e, G), G);
}

inline unsigned OptimizationBranchCompiler::getNumOfInputBindings(const Kernel * const kernel, const RelationshipType type) const {
    return (type == RelationshipType::StreamSet) ? kernel->getNumOfStreamInputs() : kernel->getNumOfScalarInputs();
}

inline const Binding & OptimizationBranchCompiler::getInputBinding(const Kernel * const kernel, const RelationshipType type, const unsigned i) const {
    return (type == RelationshipType::StreamSet) ? kernel->mInputStreamSets[i] : kernel->mInputScalars[i];
}

inline unsigned OptimizationBranchCompiler::getNumOfOutputBindings(const Kernel * const kernel, const RelationshipType type) const {
    return (type == RelationshipType::StreamSet) ? kernel->getNumOfStreamOutputs() : kernel->getNumOfScalarOutputs();
}

inline const Binding & OptimizationBranchCompiler::getOutputBinding(const Kernel * const kernel, const RelationshipType type, const unsigned i) const {
    return (type == RelationshipType::StreamSet) ? kernel->mOutputStreamSets[i] : kernel->mOutputScalars[i];
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief concat
 ** ------------------------------------------------------------------------------------------------------------- */
inline StringRef concat(StringRef A, StringRef B, SmallVector<char, 256> & tmp) {
    Twine C = A + B;
    tmp.clear();
    return C.toStringRef(tmp);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeRelationshipGraph
 ** ------------------------------------------------------------------------------------------------------------- */
RelationshipGraph OptimizationBranchCompiler::makeRelationshipGraph(const RelationshipType type) const {

    using Vertex = RelationshipGraph::vertex_descriptor;
    using Map = flat_map<const Relationship *, Vertex>;

    RelationshipGraph G(INITIAL_GRAPH_SIZE);
    Map M;

    auto addRelationship = [&](const Relationship * const rel) {
        const auto f = M.find(rel);
        if (LLVM_UNLIKELY(f != M.end())) {
            return f->second;
        } else {
            const auto x = add_vertex(G);
            M.emplace(rel, x);
            return x;
        }
    };

    const auto numOfInputs = getNumOfInputBindings(mTarget, type);
    for (unsigned i = 0; i < numOfInputs; ++i) {
        const auto & input = getInputBinding(mTarget, type, i);
        const auto r = addRelationship(input.getRelationship());
        add_edge(BRANCH_INPUT, r, RelationshipRef{i, input.getName(), input}, G);
    }

    const auto numOfOutputs = getNumOfOutputBindings(mTarget, type);
    for (unsigned i = 0; i < numOfOutputs; ++i) {
        const auto & output = getOutputBinding(mTarget, type, i);
        const auto r = addRelationship(output.getRelationship());
        add_edge(r, BRANCH_OUTPUT, RelationshipRef{i, output.getName(), output}, G);
    }

    if (type == RelationshipType::StreamSet && isa<StreamSet>(mCondition)) {
        const auto r = addRelationship(mCondition);
        add_edge(r, CONDITION_VARIABLE, RelationshipRef{}, G);
    }

    if (type == RelationshipType::Scalar && isa<Scalar>(mCondition)) {
        const auto r = addRelationship(mCondition);
        add_edge(r, CONDITION_VARIABLE, RelationshipRef{}, G);
    }

    auto linkRelationships = [&](const Kernel * const kernel, const Vertex branch) {

        auto findRelationship = [&](const Binding & binding) {
            Relationship * const rel = binding.getRelationship();
            const auto f = M.find(rel);
            if (LLVM_UNLIKELY(f == M.end())) {
                if (LLVM_LIKELY(rel->isConstant())) {
                    const auto x = add_vertex(G);
                    M.emplace(rel, x);
                    return x;
                } else {
                    std::string tmp;
                    raw_string_ostream msg(tmp);
                    msg << "Branch was not provided a ";
                    if (type == RelationshipType::StreamSet) {
                        msg << "StreamSet";
                    } else if (type == RelationshipType::Scalar) {
                        msg << "Scalar";
                    }
                    msg << " binding for "
                        << kernel->getName()
                        << '.'
                        << binding.getName();
                    report_fatal_error(StringRef(msg.str()));
                }
            }
            return f->second;
        };

        const auto numOfInputs = getNumOfInputBindings(kernel, type);
        for (unsigned i = 0; i < numOfInputs; ++i) {
            const auto & input = getInputBinding(mTarget, type, i);
            const auto r = findRelationship(input);
            add_edge(r, branch, RelationshipRef{i, input.getName(), input}, G);
        }

        const auto numOfOutputs = getNumOfOutputBindings(kernel, type);
        for (unsigned i = 0; i < numOfOutputs; ++i) {
            const auto & output = getOutputBinding(kernel, type, i);
            const auto r = findRelationship(output);
            add_edge(branch, r, RelationshipRef{i, output.getName(), output}, G);
        }
    };

    linkRelationships(mBranches[ALL_ZERO_BRANCH], ALL_ZERO_BRANCH);
    linkRelationships(mBranches[NON_ZERO_BRANCH], NON_ZERO_BRANCH);

    return G;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::addBranchProperties(KernelBuilder & b) {

    // 0 or 1 value to allow the pipeline to tell this kernel what branch to take
    mTarget->addThreadLocalScalar(b.getInt1Ty(), CONTROL_CODE);

    IntegerType * const sizeTy = b.getSizeTy();

    mTarget->addInternalScalar(sizeTy, EXTERNAL_SEGMENT_NUMBER, 0);

    mTarget->addInternalScalar(sizeTy, ALL_ZERO_PATH_TAKEN_COUNT, 0);

    mTarget->addInternalScalar(sizeTy, ALL_ZERO_INTERNAL_INFLIGHT_COUNT, 1);

    mTarget->addInternalScalar(sizeTy, NON_ZERO_INTERNAL_INFLIGHT_COUNT, 2);

    for (unsigned i = ALL_ZERO_BRANCH; i <= NON_ZERO_BRANCH; ++i) {
        const Kernel * const kernel = mBranches[i]; assert (kernel);

        if (LLVM_LIKELY(kernel->isStateful())) {
            Type * handlePtrType = nullptr;
            if (LLVM_UNLIKELY(kernel->getNumOfNestedKernelFamilyCalls())) {
                handlePtrType = b.getVoidPtrTy();
            } else {
                handlePtrType = kernel->getSharedStateType()->getPointerTo();
            }
            mTarget->addInternalScalar(handlePtrType, SHARED_PREFIX + std::to_string(i), i + 2);
        }

        if (kernel->hasThreadLocal()) {
            Type * handlePtrType = nullptr;
            if (LLVM_UNLIKELY(kernel->getNumOfNestedKernelFamilyCalls())) {
                handlePtrType = b.getVoidPtrTy();
            } else {
                handlePtrType = kernel->getThreadLocalStateType()->getPointerTo();
            }
            mTarget->addThreadLocalScalar(handlePtrType, THREAD_LOCAL_PREFIX + std::to_string(i));
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructStreamSetBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::constructStreamSetBuffers(KernelBuilder & b) {

    mStreamSetInputBuffers.clear();
    const auto numOfInputStreams = mInputStreamSets.size();
    mStreamSetInputBuffers.resize(numOfInputStreams);
    for (unsigned i = 0; i < numOfInputStreams; ++i) {
        const Binding & input = mInputStreamSets[i];
        mStreamSetInputBuffers[i].reset(new ExternalBuffer(i, b, input.getType(), 0));
    }

    mStreamSetOutputBuffers.clear();
    const auto numOfOutputStreams = mOutputStreamSets.size();
    mStreamSetOutputBuffers.resize(numOfOutputStreams);
    for (unsigned i = 0; i < numOfOutputStreams; ++i) {
        const Binding & output = mOutputStreamSets[i];
        StreamSetBuffer * buffer = nullptr;
        if (Kernel::isLocalBuffer(output).any()) {
            buffer = new ManagedDynamicBuffer(i + numOfInputStreams, b, output.getType(), true, 0);
        } else {
            buffer = new ExternalBuffer(i + numOfInputStreams, b, output.getType(), 0);
        }
        mStreamSetOutputBuffers[i].reset(buffer);
    }

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::generateInitializeMethod(KernelBuilder & b) {
    using TC = KernelBuilder::TerminationCode;
    mScalarCache.clear();
    std::vector<Value *> args;
    ConstantInt * const ZERO = b.getSize(0);
    Value * terminated = b.getFalse();
    for (unsigned i = ALL_ZERO_BRANCH; i <= NON_ZERO_BRANCH; ++i) {
        const Kernel * const kernel = mBranches[i];
        if (LLVM_UNLIKELY(kernel == nullptr)) continue;
        const bool hasSharedState = kernel->isStateful();
        const auto firstArgIndex = hasSharedState ? 1U : 0U;
        args.resize(firstArgIndex + in_degree(i, mScalarGraph));
        if (LLVM_LIKELY(hasSharedState)) {
            Value * handle = nullptr;
            if (kernel->getNumOfNestedKernelFamilyCalls()) {
                handle = b.getScalarField(SHARED_PREFIX + std::to_string(i));
            } else {
                handle = kernel->createInstance(b);
                b.setScalarField(SHARED_PREFIX + std::to_string(i), handle);
            }
            args[0] = handle;
        }
        for (const auto e : make_iterator_range(in_edges(i, mScalarGraph))) {
            const RelationshipRef & ref = mScalarGraph[e];
            const auto j = ref.Index + firstArgIndex;
            args[j] = getInputScalar(b, source(e, mScalarGraph));
        }
        Function * initFn = kernel->getInitializeFunction(b);
        FunctionType * fTy = initFn->getFunctionType();
        assert (fTy->getNumParams() == args.size());
        Value * const terminatedOnInit = b.CreateCall(fTy, initFn, args);
        Value * const term = b.CreateICmpNE(terminatedOnInit, ZERO);
        terminated = b.CreateOr(terminated, term);
    }

    Value * const termSignal = b.CreateSelect(terminated, b.getSize(TC::Fatal), b.getSize(TC::None));
    b.CreateStore(termSignal, getTerminationSignalPtr());
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::generateAllocateSharedInternalStreamSetsMethod(KernelBuilder & b, Value * const expectedNumOfStrides) {
    allocateOwnedBranchBuffers(b, expectedNumOfStrides, true);
    // allocate any owned output buffers
    const auto n = mTarget->getNumOfStreamOutputs();
    for (unsigned i = 0; i != n; ++i) {
        const Binding & output = mTarget->getOutputStreamSetBinding(i);
        if (LLVM_LIKELY(Kernel::isLocalBuffer(output).any())) {
            auto & buffer = mStreamSetOutputBuffers[i];
            assert (buffer->getHandle());
            buffer->allocateBuffer(b, expectedNumOfStrides, nullptr, nullptr, nullptr);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::generateInitializeThreadLocalMethod(KernelBuilder & b) {
    for (unsigned i = ALL_ZERO_BRANCH; i <= NON_ZERO_BRANCH; ++i) {
        const Kernel * const kernel = mBranches[i]; assert (kernel);
        if (kernel->hasThreadLocal()) {

            SmallVector<Value *, 2> args;
            Value * const shared = loadSharedHandle(b, i);
            if (shared) {
                args.push_back(shared);
            }
            args.push_back(ConstantPointerNull::get(kernel->getThreadLocalStateType()->getPointerTo()));

            Value * const handle = kernel->initializeThreadLocalInstance(b, args);
            b.setScalarField(THREAD_LOCAL_PREFIX + std::to_string(i), handle);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateThreadLocalInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::generateAllocateThreadLocalInternalStreamSetsMethod(KernelBuilder & b, Value * const expectedNumOfStrides) {
    allocateOwnedBranchBuffers(b, expectedNumOfStrides, false);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadSharedHandle
 ** ------------------------------------------------------------------------------------------------------------- */
Value * OptimizationBranchCompiler::loadSharedHandle(KernelBuilder & b, const unsigned branchType) const {
    const Kernel * const kernel = mBranches[branchType]; assert (kernel);
    Value * handle = nullptr;
    if (LLVM_LIKELY(kernel->isStateful())) {
        handle = b.getScalarField(SHARED_PREFIX + std::to_string(branchType));
        if (kernel->getNumOfNestedKernelFamilyCalls()) {
            handle = b.CreatePointerCast(handle, kernel->getSharedStateType()->getPointerTo());
        }
    }
    return handle;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadThreadLocalHandle
 ** ------------------------------------------------------------------------------------------------------------- */
Value * OptimizationBranchCompiler::loadThreadLocalHandle(KernelBuilder & b, const unsigned branchType) const {
    const Kernel * const kernel = mBranches[branchType]; assert (kernel);
    Value * handle = nullptr;
    if (LLVM_LIKELY(kernel->hasThreadLocal())) {
        handle = b.getScalarField(THREAD_LOCAL_PREFIX + std::to_string(branchType));
        if (kernel->getNumOfNestedKernelFamilyCalls()) {
            handle = b.CreatePointerCast(handle, kernel->getThreadLocalStateType()->getPointerTo());
        }
    }
    return handle;
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::generateKernelMethod(KernelBuilder & b) {

    Value * const selectedBranch = b.getScalarField(CONTROL_CODE);

    BasicBlock * const optimizationBranch = b.CreateBasicBlock("optimizationBranch");
    BasicBlock * const normalBranch = b.CreateBasicBlock("normalBranchInvoke");
    BasicBlock * const exitLoop = b.CreateBasicBlock("exitLoop");
    b.CreateCondBr(selectedBranch, normalBranch, optimizationBranch);

    b.SetInsertPoint(exitLoop);
    if (canSetTerminateSignal()) {
        mTerminatedPhi = b.CreatePHI(b.getInt1Ty(), 2);
    } else {
        mTerminatedPhi = nullptr;
    }

    b.SetInsertPoint(optimizationBranch);
    executeBranch(b, ALL_ZERO_BRANCH);
    b.CreateBr(exitLoop);

    b.SetInsertPoint(normalBranch);
    executeBranch(b, NON_ZERO_BRANCH);
    b.CreateBr(exitLoop);

    b.SetInsertPoint(exitLoop);
    if (mTerminatedPhi) {
        b.CreateStore(mTerminatedPhi, getTerminationSignalPtr());
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getConditionRef
 ** ------------------------------------------------------------------------------------------------------------- */
inline const RelationshipRef & getConditionRef(const RelationshipGraph & G) {
    return G[parent(in_edge(CONDITION_VARIABLE, G), G)];
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::executeBranch(KernelBuilder & b, const unsigned branchType) {
#if 0
    const Kernel * const kernel = mBranches[branchType];

    if (LLVM_UNLIKELY(kernel == nullptr)) {
        report_fatal_error("empty branch not supported yet.");
    }

    Value * const segNo = enterBranch(b, branchType);

    Function * const doSegment = kernel->getDoSegmentFunction(b);
    SmallVector<Value *, 64> args;
    args.reserve(doSegment->arg_size());

    auto addNextArg = [&](Value * arg) {
        #ifndef NDEBUG
        FunctionType * const funcType = cast<FunctionType>(doSegment->getType()->getPointerElementType());
        assert ("null argument" && arg);
        assert ("too many arguments?" && args.size() < funcType->getNumParams());
        assert ("invalid argument type" && (funcType->getParamType(args.size()) == arg->getType()));
        #endif
        args.push_back(arg);
    };

    if (kernel->isStateful()) {
        addNextArg(loadSharedHandle(b, branchType));
    }
    if (kernel->hasThreadLocal()) {
        addNextArg(loadThreadLocalHandle(b, branchType));
    }
    if (kernel->hasAttribute(AttrId::InternallySynchronized)) {
        addNextArg(segNo);
    }
    addNextArg(mRawNumOfStrides);
    if (kernel->hasFixedRateIO()) {
        addNextArg(mFixedRateFactor);
    }

    PointerType * const voidPtrTy = b.getVoidPtrTy();

    for (const auto e : make_iterator_range(in_edges(branchType, mStreamSetGraph))) {
        const RelationshipRef & host = mStreamSetGraph[e];
        const RelationshipRef & path = mStreamSetGraph[parent(e, mStreamSetGraph)];
        const auto & buffer = mStreamSetInputBuffers[host.Index];
        addNextArg(b.CreatePointerCast(buffer->getBaseAddress(b), voidPtrTy));
        const Binding & input = kernel->getInputStreamSetBinding(path.Index);
        Value * processed = mProcessedInputItemPtr[host.Index];
        if (isAddressable(input)) {
            addNextArg(processed);
        } else if (isCountable(input)) {
            addNextArg(b.CreateLoad(processed));
        }
        if (LLVM_UNLIKELY(requiresItemCount(input))) {
            addNextArg(getAccessibleInputItems(host.Index));
        }
    }

    const auto canTerminate = kernel->canSetTerminateSignal();

    for (const auto e : make_iterator_range(out_edges(branchType, mStreamSetGraph))) {
        const RelationshipRef & host = mStreamSetGraph[e];
        const auto & buffer = mStreamSetOutputBuffers[host.Index];
        const RelationshipRef & path = mStreamSetGraph[child(e, mStreamSetGraph)];
        const Binding & output = kernel->getOutputStreamSetBinding(path.Index);
        const auto isShared = output.hasAttribute(AttrId::SharedManagedBuffer);
        const auto isLocal =  Kernel::isLocalBuffer(output, false);

        /// ----------------------------------------------------
        /// logical buffer base address
        /// ----------------------------------------------------
        if (LLVM_UNLIKELY(isShared)) {
            Value * const handle = buffer->getHandle();
            addNextArg(b.CreatePointerCast(handle, buffer->getHandlePointerType(b)));
        } else if (LLVM_UNLIKELY(isLocal)) {
            addNextArg(mUpdatableOutputBaseVirtualAddressPtr[path.Index]);
        } else {
            Value * const vba = buffer->getBaseAddress(b);
            addNextArg(b.CreatePointerCast(vba, voidPtrTy));
        }

        /// ----------------------------------------------------
        /// produced item count
        /// ----------------------------------------------------
        Value * produced = mProducedOutputItemPtr[host.Index];
        if (LLVM_LIKELY(canTerminate || isAddressable(output))) {
            addNextArg(produced);
        } else if (LLVM_LIKELY(isCountable(output))) {
            addNextArg(b.CreateLoad(produced));
        }

        /// ----------------------------------------------------
        /// writable / consumed item count
        /// ----------------------------------------------------
        if (isShared || isLocal) {
            addNextArg(mConsumedOutputItems[host.Index]);
        } else if (requiresItemCount(output)) {
            addNextArg(mWritableOutputItems[host.Index]);
        }
    }

    Value * const terminated = b.CreateCall(doSegment->getFunctionType(), doSegment, args);

    exitBranch(b, branchType);

    if (mTerminatedPhi) {
        Value * const termSignal = canTerminate ? terminated : b.getFalse();
        mTerminatedPhi->addIncoming(termSignal, b.GetInsertBlock());
    }
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief enterBranch
 ** ------------------------------------------------------------------------------------------------------------- */
Value * OptimizationBranchCompiler::enterBranch(KernelBuilder & b, const unsigned branchType) const {
#if 0
    const Kernel * const kernel = mBranches[branchType];
    const auto prefix = kernel->getName();

    SmallVector<char, 256> tmp;
    BasicBlock * const acquire = b.CreateBasicBlock(concat(prefix, "_acquire", tmp));
    BasicBlock * const acquired = b.CreateBasicBlock(concat(prefix, "_acquired", tmp));

    Value * const externalSegNoPtr = b.getScalarFieldPtr(EXTERNAL_SEGMENT_NUMBER);
    b.CreateBr(acquire);

    b.SetInsertPoint(acquire);
    Value * const currentSegNo = b.CreateAtomicLoadAcquire(externalSegNoPtr);
    Value * const ready = b.CreateICmpEQ(getExternalSegNo(), currentSegNo);
    b.CreateLikelyCondBr(ready, acquired, acquire);

    b.SetInsertPoint(acquired);
    // While the branched kernels may be internally synchronized, they do not share
    // any synchronization betwee them. To avoid complications, we can only permit
    // one branch type to execute at a given moment.
    Value * otherBranchCountPtr = nullptr;
    if (branchType == ALL_ZERO_BRANCH) {
        otherBranchCountPtr = b.getScalarFieldPtr(NON_ZERO_INTERNAL_INFLIGHT_COUNT);
    } else {
        otherBranchCountPtr = b.getScalarFieldPtr(ALL_ZERO_INTERNAL_INFLIGHT_COUNT);
    }
    BasicBlock * const checkAlternateBranch = b.CreateBasicBlock(concat(prefix, "_checkAlternateBranch", tmp));
    BasicBlock * const executeKernel = b.CreateBasicBlock(concat(prefix, "_executeKernel", tmp));

    b.CreateBr(checkAlternateBranch);

    b.SetInsertPoint(checkAlternateBranch);
    Value * const currentCount = b.CreateAtomicLoadAcquire(otherBranchCountPtr);
    if (codegen::DebugOptionIsSet(codegen::EnableAsserts)) {
        b.CreateAssert(b.CreateICmpULE(currentCount, b.getSize(4)), "other branch bad");
    }
    Value * const safe = b.CreateICmpEQ(currentCount, b.getSize(0));
    b.CreateCondBr(safe, executeKernel, checkAlternateBranch);

    b.SetInsertPoint(executeKernel);
    // We're still protected by the outer sync lock so an atomic add is not required here.

    Value * const sz_ONE = b.getSize(1);

    Value * branchCountPtr = nullptr;
    if (branchType == ALL_ZERO_BRANCH) {
        branchCountPtr = b.getScalarFieldPtr(ALL_ZERO_INTERNAL_INFLIGHT_COUNT);
    } else {
        branchCountPtr = b.getScalarFieldPtr(NON_ZERO_INTERNAL_INFLIGHT_COUNT);
    }
    Value * const nextCount = b.CreateAdd(b.CreateLoad(branchCountPtr), sz_ONE);
    b.CreateStore(nextCount, branchCountPtr);

    Value * const intSegNoPtr = b.getScalarFieldPtr(ALL_ZERO_PATH_TAKEN_COUNT);
    Value * intSegNo = b.CreateLoad(intSegNoPtr);
    if (branchType == ALL_ZERO_BRANCH) {
        b.CreateStore(b.CreateAdd(intSegNo, sz_ONE), intSegNoPtr);
    } else {
        intSegNo = b.CreateSub(getExternalSegNo(), intSegNo);
    }

    if (LLVM_LIKELY(kernel->hasAttribute(AttrId::InternallySynchronized))) {
        Value * const released = b.CreateAdd(getExternalSegNo(), sz_ONE);
        b.CreateAtomicStoreRelease(released, externalSegNoPtr);
    }
    return intSegNo;
#endif
    return nullptr;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief exitBranch
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::exitBranch(KernelBuilder & b, const unsigned branchType) const {
#if 0
    // decrement the number of active threads count for this branch
    Constant * const sz_ONE = b.getSize(1);

    Value * branchCountPtr = nullptr;
    if (branchType == ALL_ZERO_BRANCH) {
        branchCountPtr = b.getScalarFieldPtr(ALL_ZERO_INTERNAL_INFLIGHT_COUNT);
    } else {
        branchCountPtr = b.getScalarFieldPtr(NON_ZERO_INTERNAL_INFLIGHT_COUNT);
    }
    if (LLVM_LIKELY(mBranches[branchType]->hasAttribute(AttrId::InternallySynchronized))) {
        b.CreateAtomicFetchAndSub(sz_ONE, branchCountPtr);
        if (codegen::DebugOptionIsSet(codegen::EnableAsserts)) {
            Value * const c = b.CreateAtomicLoadAcquire(branchCountPtr);
            b.CreateAssert(b.CreateICmpULE(c, b.getSize(4)), "released branch bad");
        }
    } else {
        b.CreateStore(b.CreateSub(b.CreateLoad(branchCountPtr), sz_ONE), branchCountPtr);
        Value * const externalSegNoPtr = b.getScalarFieldPtr(EXTERNAL_SEGMENT_NUMBER);
        Value * const released = b.CreateAdd(getExternalSegNo(), sz_ONE);
        b.CreateAtomicStoreRelease(released, externalSegNoPtr);
    }
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getScalar
 ** ------------------------------------------------------------------------------------------------------------- */
inline Value * OptimizationBranchCompiler::getInputScalar(KernelBuilder & b, const unsigned scalar) {
    const auto f = mScalarCache.find(scalar);
    if (LLVM_UNLIKELY(f != mScalarCache.end())) {
        return f->second;
    }
    const auto e = in_edge(scalar, mScalarGraph);
    const RelationshipRef & ref = mScalarGraph[e];
    Value * const value = b.getScalarField(ref.Name);
    mScalarCache.emplace(scalar, value);
    return value;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::generateFinalizeThreadLocalMethod(KernelBuilder & b) {
    for (unsigned i = ALL_ZERO_BRANCH; i <= NON_ZERO_BRANCH; ++i) {
        const Kernel * const kernel = mBranches[i];
        if (kernel->hasThreadLocal()) {
            SmallVector<Value *, 2> args;
            if (LLVM_LIKELY(kernel->isStateful())) {
                args.push_back(loadSharedHandle(b, i));
            }
            args.push_back(loadThreadLocalHandle(b, i));
            kernel->finalizeThreadLocalInstance(b, args);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::generateFinalizeMethod(KernelBuilder & b) {
    for (unsigned i = ALL_ZERO_BRANCH; i <= NON_ZERO_BRANCH; ++i) {
        const Kernel * const kernel = mBranches[i];
        SmallVector<Value *, 2> args;
        if (LLVM_LIKELY(kernel->isStateful())) {
            args.push_back(loadSharedHandle(b, i));
        }
        if (LLVM_LIKELY(kernel->hasThreadLocal())) {
            args.push_back(loadThreadLocalHandle(b, i));
        }
        kernel->finalizeInstance(b, args);
    }
    // allocate any owned output buffers
    const auto n = mTarget->getNumOfStreamOutputs();
    for (unsigned i = 0; i != n; ++i) {
        const Binding & output = mTarget->getOutputStreamSetBinding(i);
        if (LLVM_LIKELY(Kernel::isLocalBuffer(output).any())) {
            auto & buffer = mStreamSetOutputBuffers[i];
            assert (buffer->getHandle());
            buffer->releaseBuffer(b);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getFinalOutputScalars
 ** ------------------------------------------------------------------------------------------------------------- */
std::vector<Value *> OptimizationBranchCompiler::getFinalOutputScalars(KernelBuilder & b) {
    // TODO: IMPLEMENT THIS!
    return std::vector<Value *>{};
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief allocateOwnedBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranchCompiler::allocateOwnedBranchBuffers(KernelBuilder & b, Value * const expectedNumOfStrides, const bool nonLocal) {
    assert (expectedNumOfStrides);
    // recursively allocate any internal buffers for the nested kernels, giving them the correct
    // num of strides it should expect to perform
    for (unsigned i = ALL_ZERO_BRANCH; i <= NON_ZERO_BRANCH; ++i) {
        const Kernel * const kernelObj = mBranches[i];
        if (LLVM_UNLIKELY(kernelObj == nullptr)) continue;
        if (LLVM_UNLIKELY(kernelObj->allocatesInternalStreamSets())) {
            if (nonLocal || kernelObj->hasThreadLocal()) {
                SmallVector<Value *, 3> params;
                if (LLVM_LIKELY(kernelObj->isStateful())) {
                    params.push_back(loadSharedHandle(b, i));
                }
                Function * func = nullptr;
                if (nonLocal) {
                    func = kernelObj->getAllocateSharedInternalStreamSetsFunction(b, false);
                } else {
                    func = kernelObj->getAllocateThreadLocalInternalStreamSetsFunction(b, false);
                    params.push_back(loadThreadLocalHandle(b, i));
                }
                params.push_back(expectedNumOfStrides);
                FunctionType * fTy = func->getFunctionType();
                b.CreateCall(fTy, func, params);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeBranches
 ** ------------------------------------------------------------------------------------------------------------- */
inline std::array<const Kernel *, 4> makeBranches(const OptimizationBranch * const branch) {
    std::array<const Kernel *, 4> branches;
    branches[BRANCH_INPUT] = branch;
    branches[ALL_ZERO_BRANCH] = branch->getAllZeroKernel();
    branches[NON_ZERO_BRANCH] = branch->getNonZeroKernel();
    branches[BRANCH_OUTPUT] = branch;

    for (unsigned i = ALL_ZERO_BRANCH; i <= NON_ZERO_BRANCH; ++i) {
        const Kernel * const kernel = branches[i];
        if (LLVM_UNLIKELY(!kernel->hasAttribute(AttrId::InternallySynchronized))) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "Branch " << kernel->getName() << " of "
                   "OptimizationBranch " << branch->getName() <<
                   " must be InternallySynchronized.";
            report_fatal_error(StringRef(out.str()));
        }
    }

    return branches;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
OptimizationBranchCompiler::OptimizationBranchCompiler(KernelBuilder & b, OptimizationBranch * const branch) noexcept
: KernelCompiler(branch)
, mCondition(branch->getCondition())
, mBranches(makeBranches(branch))
, mStreamSetGraph(makeRelationshipGraph(RelationshipType::StreamSet))
, mScalarGraph(makeRelationshipGraph(RelationshipType::Scalar)) {

}

}

