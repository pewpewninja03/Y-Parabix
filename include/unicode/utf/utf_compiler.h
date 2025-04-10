#pragma once

#include <unicode/core/UCD_Config.h>
#include <unicode/core/unicode_set.h>
#include <unicode/utf/utf_encoder.h>
#include <vector>
#include <pablo/pablo_toolchain.h>

namespace re {
    class CC;
}

namespace pablo {
    class PabloBuilder;
    class PabloAST;
    class Var;
}

namespace UTF {

// Kernel annotation string based on command line parameters
// controlling the UTF if-hierarchy.
std::string kernelAnnotation();

using Target_List = std::vector<pablo::Var *>;
using CC_List = std::vector<UCD::UnicodeSet>;

class UTF_Compiler {
public:
    UTF_Compiler(pablo::Var * basisVar, pablo::PabloBuilder & pb,
                 pablo::PabloAST * mask = nullptr,
                 pablo::BitMovementMode m = pablo::BitMovementMode::Advance);
    UTF_Compiler(pablo::Var * basisVar, pablo::PabloBuilder & pb,
                 pablo::BitMovementMode m);
    void compile(Target_List targets, CC_List ccs);
    void compile(Target_List targets, std::vector<re::CC *> ccs);
protected:
    pablo::Var *            mVar;
    pablo::PabloBuilder &   mPB;
    pablo::PabloAST *       mMask;
    pablo::BitMovementMode  mBitMovement;
};

}
