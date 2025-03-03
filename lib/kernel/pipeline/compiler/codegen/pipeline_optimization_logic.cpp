#include "../pipeline_compiler.hpp"
#ifndef NDEBUG
#include <llvm/IR/Verifier.h>
#endif

namespace kernel {


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief replacePhiCatchBlocksWith
 *
 * replace the phi catch with the actual exit blocks
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::replacePhiCatchWithCurrentBlock(KernelBuilder & b, BasicBlock *& toReplace, BasicBlock * const phiContainer) {
    // NOTE: not all versions of LLVM seem to have BasicBlock::replacePhiUsesWith or PHINode::replaceIncomingBlockWith.
    // This code could be made to use those instead.

    assert (toReplace);

    BasicBlock * const to = b.GetInsertBlock();

    for (Instruction & inst : *phiContainer) {
        if (LLVM_LIKELY(isa<PHINode>(inst))) {
            PHINode & pn = cast<PHINode>(inst);
            for (unsigned i = 0; i != pn.getNumIncomingValues(); ++i) {
                if (pn.getIncomingBlock(i) == toReplace) {
                    pn.setIncomingBlock(i, to);
                }
            }
        } else {
            break;
        }
    }

    if (!toReplace->empty()) {
        Instruction * toMove = &toReplace->front();
        while (toMove) {
            Instruction * const next = toMove->getNextNode();
            toMove->removeFromParent();
            toMove->insertAfter(&to->back());
            toMove = next;
        }
    }


    toReplace->eraseFromParent();
    toReplace = to;

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runOptimizationPasses
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addOptimizationPasses(KernelBuilder & b, Kernel::SelectedOptimizationPasses & passes) {

    // To make sure the optimizations aren't hiding an error, first run the verifier
    // detect any possible errors prior to optimizing it.

    using P = Kernel::OptimizationPass;

    #ifndef NDEBUG
    Module * const m = b.getModule();
    SmallVector<char, 256> tmp;
    raw_svector_ostream msg(tmp);
    bool BrokenDebugInfo = false;
    if (LLVM_UNLIKELY(verifyModule(*m, &msg, &BrokenDebugInfo))) {
        m->print(errs(), nullptr);
        report_fatal_error(StringRef(msg.str()));
    }
    #endif

    passes.push_back(P::PHICanonicalizerPass);
    passes.push_back(P::DCEPass);
    passes.push_back(P::SimplifyCFGPass);
    passes.push_back(P::PHICanonicalizerPass);
    passes.push_back(P::EarlyCSEPass);
    passes.push_back(P::MemCpyOptPass);

}

}
