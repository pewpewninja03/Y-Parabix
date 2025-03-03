#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/pipeline/optimizationbranch.h>
#include <kernel/core/kernel_builder.h>
#include <boost/container/flat_map.hpp>
#include <boost/iterator/function_output_iterator.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Timer.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Constants.h>
#include <toolchain/toolchain.h>
#include "compiler/pipeline_compiler.hpp"

// TODO: the builders should detect if there is only one kernel in a pipeline / both branches are equivalent and return the single kernel. Modify addOrDeclareMainFunction.

// TODO: make a templated compile method to automatically validate and cast the main function to the correct type

using namespace llvm;
using namespace boost;
using namespace boost::container;

template <typename IntTy>
inline IntTy round_up_to(const IntTy x, const IntTy y) {
    return (x + y - 1) & -y;
}

#define ADD_CL_SCALAR(Id,Type) \
    mTarget->mInputScalars.emplace_back(Id, mDriver.CreateCommandLineScalar(CommandLineScalarType::Type))

namespace kernel {

using Scalars = PipelineKernel::Scalars;

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernel
 ** ------------------------------------------------------------------------------------------------------------- */
Kernel * PipelineBuilder::initializeKernel(Kernel * const kernel, const unsigned flags) {
    mDriver.addKernel(kernel);
    mTarget->mKernels.emplace_back(kernel, flags);
    return kernel;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateRepeatingBixNum
 ** ------------------------------------------------------------------------------------------------------------- */
StreamSet * PipelineBuilder::CreateRepeatingBixNum(unsigned bixNumBits, pattern_t nums, bool isDynamic) {
    std::vector<std::vector<uint64_t>> templatePattern;
    templatePattern.resize(bixNumBits);
    for (unsigned i = 0; i < bixNumBits; i++) {
        templatePattern[i].resize(nums.size());
    }
    for (unsigned j = 0; j < nums.size(); j++) {
        for (unsigned i = 0; i < bixNumBits; i++) {
            templatePattern[i][j] = static_cast<uint64_t>((nums[j] >> i) & 1U);
        }
    }
    return CreateRepeatingStreamSet(1, templatePattern, isDynamic);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief captureBitstream
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineBuilder::captureBitstream(llvm::StringRef streamName, StreamSet * bitstream, char zeroCh, char oneCh) {
    mTarget->mIllustratorBindings.emplace_back(IllustratorTypeId::Bitstream, streamName.str(), bitstream, zeroCh, oneCh);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief captureBixNum
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineBuilder::captureBixNum(llvm::StringRef streamName, StreamSet * bixnum, char hexBase) {
    mTarget->mIllustratorBindings.emplace_back(IllustratorTypeId::BixNum, streamName.str(), bixnum, hexBase);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief captureByteData
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineBuilder::captureByteData(llvm::StringRef streamName, StreamSet * byteData, char nonASCIIsubstitute) {
    mTarget->mIllustratorBindings.emplace_back(IllustratorTypeId::ByteData, streamName.str(), byteData, nonASCIIsubstitute);
}

using Kernels = PipelineBuilder::Kernels;

enum class VertexType { Kernel, StreamSet, Scalar };

using AttrId = Attribute::KindId;

using TypeId = Relationship::ClassTypeId;

using Graph = adjacency_list<hash_setS, vecS, bidirectionalS, const Relationship *, int>;

using Vertex = Graph::vertex_descriptor;
using Map = flat_map<const Relationship *, Vertex>;

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addAttributesFrom
 *
 * Add any attributes from a set of kernels
 ** ------------------------------------------------------------------------------------------------------------- */
void addKernelProperties(const Kernels & kernels, Kernel * const output) {
    bool mustTerminate = false;
    bool canTerminate = false;
    bool sideEffecting = false;
    bool fatalTermination = false;
    unsigned stride = 0;
    for (const auto & K : kernels) {
        Kernel * kernel = K.Object;
        for (const Attribute & attr : kernel->getAttributes()) {
            switch (attr.getKind()) {
                case AttrId::MustExplicitlyTerminate:
                    mustTerminate = true;
                    break;
                case AttrId::MayFatallyTerminate:
                    fatalTermination = true;
                    break;
                case AttrId::CanTerminateEarly:
                    canTerminate = true;
                    break;
                case AttrId::SideEffecting:
                    sideEffecting = true;
                    break;
                default: continue;
            }
        }
        assert (kernel->getStride());
        if (stride) {
            stride = boost::lcm(stride, kernel->getStride());
        } else {
            stride = kernel->getStride();
        }
    }

    if (fatalTermination) {
        output->addAttribute(MayFatallyTerminate());
    }
    if (mustTerminate) {
        output->addAttribute(MustExplicitlyTerminate());
    } else if (canTerminate && !fatalTermination) {
        output->addAttribute(CanTerminateEarly());
    }
    if (sideEffecting) {
        output->addAttribute(SideEffecting());
    }
    output->setStride(stride);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeKernel
 ** ------------------------------------------------------------------------------------------------------------- */
Kernel * PipelineBuilder::makeKernel() {

    mDriver.generateUncachedKernels();

    const auto & kernels = mTarget->mKernels;
    const auto & calls = mTarget->mCallBindings;

    const auto numOfKernels = kernels.size();
    const auto numOfCalls = calls.size();

    auto & signature = mTarget->mSignature;

    unsigned numOfNestedKernelFamilyCalls = 0;

    auto & internallyGeneratedStreamSets = mTarget->mInternallyGeneratedStreamSets;

    auto addInternallyGenerated = [&](const Relationship * r) {
        assert (isa<RepeatingStreamSet>(r));
        for (unsigned i = 0; i < internallyGeneratedStreamSets.size(); ++i) {
            if (internallyGeneratedStreamSets[i] == r) {
                return;
            }
        }
        internallyGeneratedStreamSets.push_back(r);
    };

    #ifdef ENABLE_PAPI
    const auto & S = codegen::PapiCounterOptions;
    if (LLVM_UNLIKELY(S.compare(codegen::OmittedOption) != 0)) {
        ADD_CL_SCALAR(STATISTICS_PAPI_EVENT_SET_LIST, PAPIEventList);
    }
    #endif

    if (LLVM_LIKELY(signature.empty())) {

        constexpr auto pipelineInput = 0U;
        constexpr auto firstKernel = 1U;
        const auto firstCall = firstKernel + numOfKernels;
        const auto pipelineOutput = firstCall + numOfCalls;

        Graph G(pipelineOutput + 1);
        Map M;

        auto addAndMapInternallyGenerated = [&](const Relationship * r) -> unsigned {
            assert (isa<RepeatingStreamSet>(r));
            const auto f = M.find(r);
            if (LLVM_LIKELY(f == M.end())) {
                const auto v = add_vertex(r, G);
                M.emplace(r, v);
                addInternallyGenerated(r);
                return v;
            } else {
                return f->second;
            }
        };

        signature.reserve(4096);
        raw_string_ostream out(signature);

        out << 'P';
        if (mExternallySynchronized) {
            out << 'E';
        }
        for (unsigned i = 0; i < numOfKernels; ++i) {
            out << '_';
            const auto & K = kernels[i];
            auto obj = K.Object;
            obj->ensureLoaded();
            if (K.isFamilyCall()) {
                // this differs for icgrep kernel?
                out << 'F' << obj->getFamilyName();
                numOfNestedKernelFamilyCalls++;
            } else {
                out << 'K';
                const auto m = obj->getNumOfNestedKernelFamilyCalls();
                numOfNestedKernelFamilyCalls += m;
                if (obj->hasSignature()) {
                    out << obj->getSignature();
                } else {
                    out << obj->getName();
                }
            }
            if (LLVM_UNLIKELY(obj->hasInternallyGeneratedStreamSets())) {
                const auto & S = obj->getInternallyGeneratedStreamSets();
                for (size_t k = 0; k < S.size(); ++k) {
                    const auto j = addAndMapInternallyGenerated(S[k]);
                    add_edge((firstKernel + i), j, -((int)k + 1), G);
                }
            }
        }

        auto enumerateProducerBindings = [&](const Vertex producer, const Bindings & bindings) {
            const auto n = bindings.size();
            for (unsigned i = 0; i < n; ++i) {
                Relationship * const rel = bindings[i].getRelationship();
                const auto f = M.find(rel);
                if (LLVM_UNLIKELY(f != M.end())) {
                    SmallVector<char, 256> tmp;
                    raw_svector_ostream out(tmp);
                    const auto existingProducer = target(out_edge(f->second, G), G);
                    out << bindings[i].getName() << " is ";
                    if (LLVM_UNLIKELY(existingProducer == pipelineInput)) {
                        out << "an input to the pipeline";
                    } else {
                        const auto & K = kernels[existingProducer - firstKernel];
                        out << "produced by " << K.Object->getName();
                    }
                    out << " and ";
                    if (LLVM_UNLIKELY(producer == pipelineOutput)) {
                        out << "an output of the pipeline";
                    } else {
                        const auto & K = kernels[producer - firstKernel];
                        out << "produced by " << K.Object->getName();
                    }
                    out << ".";
                    report_fatal_error(StringRef(out.str()));
                }

                const auto bufferVertex = add_vertex(rel, G);
                M.emplace(rel, bufferVertex);
                add_edge(bufferVertex, producer, i, G); // buffer -> producer ordering
            }
        };

        enumerateProducerBindings(pipelineInput, mTarget->mInputScalars);
        enumerateProducerBindings(pipelineInput, mTarget->mInputStreamSets);
        for (unsigned i = 0; i < numOfKernels; ++i) {
            const auto & k = kernels[i].Object;
            k->ensureLoaded();
            enumerateProducerBindings(firstKernel + i, k->getOutputScalarBindings());
            enumerateProducerBindings(firstKernel + i, k->getOutputStreamSetBindings());
        }

        struct RelationshipVector {
            size_t size() const {
                if (LLVM_LIKELY(mUseBindings)) {
                    return mBindings->size();
                } else {
                    return mArgs->size();
                }
            }

            Relationship * getRelationship(unsigned i) const {
                if (LLVM_LIKELY(mUseBindings)) {
                    return (*mBindings)[i].getRelationship();
                } else {
                    return (*mArgs)[i];
                }
            }

            RelationshipVector(const Bindings & bindings)
            : mUseBindings(true)
            , mBindings(&bindings) {

            }

            RelationshipVector(const Scalars & args)
            : mUseBindings(false)
            , mArgs(&args) {

            }

        private:
            const bool mUseBindings;
            union {
            const Bindings * const mBindings;
            const Scalars * const mArgs;
            };
        };

        auto enumerateConsumerBindings = [&](const Vertex consumerVertex, const RelationshipVector array) {
            const auto n = array.size();
            for (unsigned i = 0; i < n; ++i) {
                Relationship * const rel = array.getRelationship(i);
                assert ("relationship cannot be null!" && rel);
                auto f = M.find(rel);
                if (LLVM_UNLIKELY(f == M.end())) {
                    if (LLVM_UNLIKELY(isa<RepeatingStreamSet>(rel) || isa<ScalarConstant>(rel) || isa<CommandLineScalar>(rel))) {
                        const auto bufferVertex = add_vertex(rel, G);
                        f = M.emplace(rel, bufferVertex).first;
                    } else {
                        SmallVector<char, 256> tmp;
                        raw_svector_ostream out(tmp);
                        if (consumerVertex < firstCall) {
                            const auto & K = kernels[consumerVertex - firstKernel];
                            const Kernel * const consumer = K.Object;
                            const Binding & input = ((rel->getClassTypeId() == TypeId::Scalar)
                                                       ? consumer->getInputScalarBinding(i)
                                                       : consumer->getInputStreamSetBinding(i));
                            out << "input " << i << " (" << input.getName() << ") of ";
                            out << "kernel " << consumer->getName();
                        } else { // TODO: function calls should retain name
                            out << "argument " << i << " of ";
                            out << "function call " << (consumerVertex - kernels.size() + 1);
                        }
                        out << " is not a constant, produced by a kernel or an input to the pipeline";
                        report_fatal_error(StringRef(out.str()));
                    }
                }
                const auto bufferVertex = f->second;
                assert (bufferVertex < num_vertices(G));
                add_edge(consumerVertex, bufferVertex, i, G); // consumer -> buffer ordering
            }
        };

        for (unsigned i = 0; i < numOfKernels; ++i) {
            Kernel * const k = kernels[i].Object;
            enumerateConsumerBindings(firstKernel + i, k->getInputScalarBindings());
            enumerateConsumerBindings(firstKernel + i, k->getInputStreamSetBindings());
        }
        for (unsigned i = 0; i < numOfCalls; ++i) {
            enumerateConsumerBindings(firstCall + i, calls[i].Args);
        }
        enumerateConsumerBindings(pipelineOutput, mTarget->mOutputScalars);
        enumerateConsumerBindings(pipelineOutput, mTarget->mOutputStreamSets);

        for (unsigned i = 0; i < numOfCalls; ++i) {
            out << "_C" << calls[i].Name;
        }
        out << '@';

        const auto firstRelationship = pipelineOutput + 1;
        const auto lastRelationship = num_vertices(G);
        int numInternallyGenerated = 0;

        using TypeId = Relationship::ClassTypeId;

        std::array<char, (unsigned)TypeId::__Count> typeCode;
        typeCode[(unsigned)TypeId::StreamSet] = 'S';
        typeCode[(unsigned)TypeId::RepeatingStreamSet] = 'R';
        typeCode[(unsigned)TypeId::TruncatedStreamSet] = 'T';
        typeCode[(unsigned)TypeId::Scalar] = 'v';
        typeCode[(unsigned)TypeId::CommandLineScalar] = 'l';
        typeCode[(unsigned)TypeId::ScalarConstant] = 'c';

        for (auto i = firstRelationship; i != lastRelationship; ++i) {
            const Relationship * const r = G[i]; assert (r);

            assert ((unsigned)r->getClassTypeId() < (unsigned)TypeId::__Count);

            out << typeCode[(unsigned)r->getClassTypeId()];

            if (LLVM_UNLIKELY(isa<RepeatingStreamSet>(r))) {
                const RepeatingStreamSet * rs = cast<RepeatingStreamSet>(r);
                if (rs->isUnaligned()) {
                    out << 'U';
                }
                if (LLVM_LIKELY(rs->isDynamic())) {
                    const auto j = addAndMapInternallyGenerated(r);
                    add_edge(j, pipelineInput, --numInternallyGenerated, G);
                } else {
                    const auto numElements = rs->getNumElements();
                    const auto fieldWidth = rs->getFieldWidth();
                    const auto width = fieldWidth / 4UL;
                    SmallVector<char, 16> tmp(width + 1);
                    for (unsigned i = 0;;) {
                        assert (i < numElements);
                        const auto & vec = rs->getPattern(i);
                        out << ':';
                        // write to hex code
                        for (auto v : vec) {
                            unsigned j = 0;
                            while (v) {
                                const auto c = (v & 15);
                                v >>= 4;
                                if (c < 10) {
                                    tmp[j] = '0' + c;
                                } else {
                                    tmp[j] = 'A' + (c - 10U);
                                }
                                ++j;
                            }
                            for (auto k = j; k < width; ++k) {
                                out << '0';
                            }
                            while (j) {
                                out << tmp[--j];
                            }
                        }
                        if ((++i) == numElements) {
                            break;
                        }
                    }
                }
                out << ':';
            } else if (LLVM_UNLIKELY(isa<TruncatedStreamSet>(r))) {
                auto f = M.find(cast<TruncatedStreamSet>(r)->getData());
                if (LLVM_UNLIKELY(f == M.end())) {
                    report_fatal_error("Truncated streamset data has no producer");
                }
                out << f->second << ':';
            }

            if (LLVM_LIKELY(out_degree(i, G) != 0)) {
                const auto e = out_edge(i, G);
                const auto j = target(e, G);
                out << j << '.' << G[e];
            }

            for (const auto e : make_iterator_range(in_edges(i, G))) {
                const auto k = source(e, G);
                out << '_' << k << '.' << G[e];
            }
        }

        out.flush();

    } else { // the programmer provided a unique name

        for (unsigned i = 0; i < numOfKernels; ++i) {
            const auto & K = kernels[i];
            Kernel * const obj = K.Object;
            obj->ensureLoaded();
            if (K.isFamilyCall()) {
                numOfNestedKernelFamilyCalls++;
            } else {
                numOfNestedKernelFamilyCalls += obj->getNumOfNestedKernelFamilyCalls();
            }
            if (LLVM_UNLIKELY(obj->hasInternallyGeneratedStreamSets())) {
                for (const auto r : obj->getInternallyGeneratedStreamSets()) {
                    addInternallyGenerated(r);
                }
            }
        }
    }

    mTarget->mNumOfKernelFamilyCalls = numOfNestedKernelFamilyCalls;

    if (mExternallySynchronized) {
        mTarget->addAttribute(InternallySynchronized());
    }

    addKernelProperties(kernels, mTarget);

    signature = PipelineKernel::annotateSignatureWithPipelineFlags(std::move(signature));

    mTarget->mKernelName =
        Kernel::annotateKernelNameWithDebugFlags(Kernel::TypeId::Pipeline,
            PipelineKernel::makePipelineHashName(signature));

    mTarget->setCompilationStatus(Kernel::CompilationStatus::FullyInitialized);

    return mTarget;
}

using AttributeCombineSet = flat_map<AttrId, unsigned>;

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief combineAttributes
 ** ------------------------------------------------------------------------------------------------------------- */
void combineAttributes(const Binding & S, AttributeCombineSet & C) {
    for (const Attribute & s : S) {
        auto f = C.find(s.getKind());
        if (LLVM_LIKELY(f == C.end())) {
            C.emplace(s.getKind(), s.amount());
        } else {
            // TODO: we'll need some form of attribute combination function
            f->second = std::max<unsigned>(f->second, s.amount());
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeKernel
 ** ------------------------------------------------------------------------------------------------------------- */
Kernel * OptimizationBranchBuilder::makeKernel() {
    llvm_unreachable("todo");
#if 0
    // TODO: the rates of the optimization branches should be determined by
    // the actual kernels within the branches.

    mNonZeroBranch->setExternallySynchronized(true);
    Kernel * const nonZero = mNonZeroBranch->makeKernel();
    if (nonZero) mDriver.addKernel(nonZero);

    mAllZeroBranch->setExternallySynchronized(true);
    Kernel * const allZero = mAllZeroBranch->makeKernel();
    if (allZero) mDriver.addKernel(allZero);

    std::string name;
    raw_string_ostream out(name);

    out << "OB:";
    if (LLVM_LIKELY(isa<StreamSet>(mCondition))) {
        StreamSet * const streamSet = cast<StreamSet>(mCondition);
        out << streamSet->getFieldWidth() << "x" << streamSet->getNumElements();
    } else {
        Scalar * const scalar = cast<Scalar>(mCondition);
        out << scalar->getFieldWidth();
    }
    out << ";Z=\"";
    if (nonZero) out << nonZero->getFamilyName();
    out << "\";N=\"";
    if (allZero) out << allZero->getFamilyName();
    out << "\"";
    out.flush();

    OptimizationBranch * const br =
            new OptimizationBranch(mDriver.getBuilder(), std::move(name),
                                   mCondition, nonZero, allZero,
                                   std::move(mInputStreamSets), std::move(mOutputStreamSets),
                                   std::move(mInputScalars), std::move(mOutputScalars));
    addKernelProperties({{nonZero, PipelineKernel::Family}, {allZero, PipelineKernel::Family}}, br);
    br->addAttribute(InternallySynchronized());
    return br;
#endif
}

StreamSet * PipelineBuilder::getInputStreamSet(const StringRef name) {
    for (Binding & input : mTarget->mInputStreamSets) {
        assert (input.getRelationship());
        if (name.compare(input.getName()) == 0) {
            return cast<StreamSet>(input.getRelationship());
        }
    }
    report_fatal_error(StringRef("no scalar named ") + name);
}

StreamSet * PipelineBuilder::getOutputStreamSet(const StringRef name) {
    for (Binding & output : mTarget->mOutputStreamSets) {
        assert (output.getRelationship());
        if (name.compare(output.getName()) == 0) {
            return cast<StreamSet>(output.getRelationship());
        }
    }
    report_fatal_error(StringRef("no scalar named ") + name);
}

Scalar * PipelineBuilder::getInputScalar(const StringRef name) {
    for (Binding & input : mTarget->mInputScalars) {
        assert (input.getRelationship());
        if (name.compare(input.getName()) == 0) {
            return cast<Scalar>(input.getRelationship());
        }
    }
    report_fatal_error(StringRef("no scalar named ") + name);
}

Scalar * PipelineBuilder::getOutputScalar(const StringRef name) {
    for (Binding & output : mTarget->mOutputScalars) {
        assert (output.getRelationship());
        if (name.compare(output.getName()) == 0) {
            return cast<Scalar>(output.getRelationship());
        }
    }
    report_fatal_error(StringRef("no scalar named ") + name);
}

void PipelineBuilder::setOutputScalar(const StringRef name, Scalar * value) {
    for (Binding & output : mTarget->mOutputScalars) {
        if (name.compare(output.getName()) == 0) {
            output.setRelationship(value);
            return;
        }
    }
    report_fatal_error(StringRef("no scalar named ") + name);
}

PipelineBuilder::PipelineBuilder(BaseDriver & driver, PipelineKernel * const kernel)
: mDriver(driver)
, mTarget(kernel) {

    auto & A = mTarget->mInputScalars;
    for (unsigned i = 0; i < A.size(); i++) {
        Binding & input = A[i];
        assert (input.getRelationship());
        if (input.getRelationship() == nullptr) {
            input.setRelationship(driver.CreateScalar(input.getType()));
        }
    }
    auto & B = mTarget->mInputStreamSets;
    for (unsigned i = 0; i < B.size(); i++) {
        Binding & input = B[i];
        assert (input.getRelationship());
        if (LLVM_UNLIKELY(input.getRelationship() == nullptr)) {
            report_fatal_error(StringRef(input.getName()) + " must be set upon construction");
        }
    }
    auto & C = mTarget->mOutputStreamSets;
    for (unsigned i = 0; i < C.size(); i++) {
        Binding & output = C[i];
        assert (output.getRelationship());
        if (LLVM_UNLIKELY(output.getRelationship() == nullptr)) {
            report_fatal_error(StringRef(output.getName()) + " must be set upon construction");
        }
    }
    auto & D = mTarget->mOutputScalars;
    for (unsigned i = 0; i < D.size(); i++) {
        Binding & output = D[i];
        assert (output.getRelationship());
        if (output.getRelationship() == nullptr) {
            output.setRelationship(driver.CreateScalar(output.getType()));
        }
    }

}

Bindings replaceManagedWithSharedManagedBuffers(const Bindings & bindings) {
    Bindings replaced;
    replaced.reserve(bindings.size());
    for (const Binding & binding : bindings) {        
        Binding newBinding(binding.getName(), binding.getRelationship(), binding.getRate());
        for (const Attribute & attr : binding.getAttributes()) {
            if (attr.getKind() == Attribute::KindId::ManagedBuffer) {
                newBinding.push_back(SharedManagedBuffer());
            } else {
                newBinding.push_back(attr);
            }
        }
        replaced.emplace_back(std::move(newBinding));
    }
    return replaced;
}

std::shared_ptr<OptimizationBranchBuilder> PipelineBuilder::CreateOptimizationBranch (
        Relationship * const condition,
        Bindings && stream_inputs, Bindings && stream_outputs,
        Bindings && scalar_inputs, Bindings && scalar_outputs) {


    auto nonZeroStreamInputs = stream_inputs;
    auto nonZeroStreamOutputs = replaceManagedWithSharedManagedBuffers(stream_outputs);
    auto nonZeroScalarInputs = scalar_inputs;
    auto nonZeroScalarOutputs = scalar_outputs;

    auto allZeroStreamInputs = nonZeroStreamInputs;
    auto allZeroStreamOutputs = nonZeroStreamOutputs;
    auto allZeroScalarInputs = nonZeroScalarInputs;
    auto allZeroScalarOutputs = nonZeroScalarOutputs;

    PipelineKernel * const allZero =
        new PipelineKernel(mDriver, "", {},
                           std::move(allZeroStreamInputs), std::move(allZeroStreamOutputs),
                           std::move(allZeroScalarInputs), std::move(allZeroScalarOutputs));

    PipelineKernel * const nonZero =
        new PipelineKernel(mDriver, "", {},
                           std::move(nonZeroStreamInputs), std::move(nonZeroStreamOutputs),
                           std::move(nonZeroScalarInputs), std::move(nonZeroScalarOutputs));

    PipelineKernel * const branch =
        new PipelineKernel(mDriver, "", {},
                           std::move(stream_inputs), std::move(stream_outputs),
                           std::move(scalar_inputs), std::move(scalar_outputs));


    std::shared_ptr<OptimizationBranchBuilder> obb(
        new OptimizationBranchBuilder(mDriver, condition, branch, allZero, nonZero));
    return obb;
}



OptimizationBranchBuilder::OptimizationBranchBuilder(
      BaseDriver & driver,
      Relationship * const condition,
      PipelineKernel * const allZero,
      PipelineKernel * const nonZero,
      PipelineKernel * const branch)
: PipelineBuilder(driver, branch)
, mCondition(condition)
, mNonZeroBranch(std::unique_ptr<PipelineBuilder>(new PipelineBuilder(mDriver, nonZero)))
, mAllZeroBranch(std::unique_ptr<PipelineBuilder>(new PipelineBuilder(mDriver, allZero))) {

}

OptimizationBranchBuilder::~OptimizationBranchBuilder() {

}

}
