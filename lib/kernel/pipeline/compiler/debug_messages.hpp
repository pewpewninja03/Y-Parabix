#include "pipeline_compiler.hpp"

#if defined(PRINT_DEBUG_MESSAGES) && !defined(DEBUG_MESSAGES_HPP)
#define DEBUG_MESSAGES_HPP

namespace kernel {

#define NEW_FILE (O_WRONLY | O_APPEND | O_CREAT | O_EXCL)

#define APPEND_FILE (O_WRONLY | O_APPEND)

#define MODE (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH)

inline void PipelineCompiler::debugInit(KernelBuilder & b) {
    Function * const pthreadSelfFn = b.getModule()->getFunction("pthread_self");
    mThreadId = b.CreateCall(pthreadSelfFn->getFunctionType(), pthreadSelfFn, {});
}

template <typename ... Args>
BOOST_NOINLINE void PipelineCompiler::debugPrint(KernelBuilder & b, Twine format, Args ...args) const {
    #ifdef PRINT_DEBUG_MESSAGES_FOR_NESTED_PIPELINE_ONLY
    if (!mIsNestedPipeline) return;
    #endif
    #ifdef PRINT_DEBUG_MESSAGES_FOR_NON_NESTED_PIPELINE_ONLY
    if (mIsNestedPipeline) return;
    #endif 
    #ifdef PRINT_DEBUG_MESSAGES_FOR_KERNEL_NUM
    const std::initializer_list<unsigned> L{0,PRINT_DEBUG_MESSAGES_FOR_KERNEL_NUM};
    if (std::find(L.begin(), L.end(), mKernelId) == L.end()) {
        return;
    }
    #endif
    #ifdef PRINT_DEBUG_MESSAGES_FOR_KERNEL_NAME
    if (mKernel == nullptr || mKernel->getName().compare(PRINT_DEBUG_MESSAGES_FOR_KERNEL_NAME) != 0) return;
    #endif
    #ifdef PRINT_DEBUG_MESSAGES_FOR_MARKED_KERNELS_ONLY
    if (LLVM_LIKELY(mKernel == nullptr || !mKernel->hasEnabledPipelineDebugMessages())) return;
    #endif
    SmallVector<char, 512> tmp;
    raw_svector_ostream out(tmp);
    #ifdef PRINT_DEBUG_MESSAGES_INCLUDE_THREAD_NUM
    if (mThreadId) {
        out << "%016" PRIx64 "  ";
    }
    #endif
    out << format << "\n";
    SmallVector<Value *, 8> argVals(2);
    argVals[0] = b.getInt32(STDERR_FILENO);
    argVals[1] = b.GetString(out.str());
    #ifdef PRINT_DEBUG_MESSAGES_INCLUDE_THREAD_NUM
    if (mThreadId) {
        argVals.push_back(mThreadId);
    }
    #endif
    argVals.append(std::initializer_list<llvm::Value *>{std::forward<Args>(args)...});
    #ifndef NDEBUG
    for (Value * arg : argVals) {
        assert ("null argument given to debugPrint" && arg);
    }
    #endif
    Function * Dprintf = b.GetDprintf();
    b.CreateCall(Dprintf->getFunctionType(), Dprintf, argVals);
}

#undef NEW_FILE
#undef APPEND_FILE
#undef MODE

}

#endif // DEBUG_MESSAGES_HPP
