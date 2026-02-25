/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/unicode/UCD_property_kernel.h>

#include <kernel/core/kernel.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <unicode/data/PropertyObjects.h>
#include <re/unicode/resolve_properties.h>
#include <unicode/utf/utf_compiler.h>
#include <kernel/core/kernel_builder.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <llvm/Support/ErrorHandling.h>

using namespace kernel;
using namespace pablo;
using namespace cc;


UnicodePropertyKernelBuilder::UnicodePropertyKernelBuilder(LLVMTypeSystemInterface & ts, re::Name * property_value_name, StreamSet * Source, StreamSet * property, pablo::BitMovementMode mode)
: UnicodePropertyKernelBuilder(ts, llvm::cast<re::PropertyExpression>(property_value_name->getDefinition()), Source, property, mode) {

}

UnicodePropertyKernelBuilder::UnicodePropertyKernelBuilder(LLVMTypeSystemInterface & ts, re::PropertyExpression * pe, StreamSet * Source, StreamSet * property, pablo::BitMovementMode mode)
: UnicodePropertyKernelBuilder(ts, pe, Source, property, mode, [&]() -> std::string {
    return std::to_string(Source->getNumElements()) +
           "x" + std::to_string(Source->getFieldWidth()) +
            pe->getFullName() +
            UTF::kernelAnnotation() +
            pablo::BitMovementMode_string(mode);
}()) {

}

UnicodePropertyKernelBuilder::UnicodePropertyKernelBuilder(LLVMTypeSystemInterface & ts, re::PropertyExpression * pe, StreamSet * Source, StreamSet * property, pablo::BitMovementMode mode, std::string && propValueName)
: PabloKernel(ts,
"UCD:" + getStringHash(propValueName) ,
{Binding{"source", Source, FixedRate(1), LookAhead(3)}},
{Binding{"property_stream", property}})
, mPropNameValue(propValueName)
, mPropertyExpr(pe)
, mBitMovement(mode) {
}

llvm::StringRef UnicodePropertyKernelBuilder::getSignature() const {
    return llvm::StringRef{mPropNameValue};
}

void UnicodePropertyKernelBuilder::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb, mBitMovement);
    pablo::Var * propertyVar = pb.createVar(mPropertyExpr->getFullName(), pb.createZeroes());
    if (mPropertyExpr->getKind() == re::PropertyExpression::Kind::Codepoint) {
        re::CC * propertyCC = llvm::cast<re::CC>(mPropertyExpr->getResolvedRE());
        unicodeCompiler.compile({propertyVar}, {propertyCC});
    } else {
        llvm::report_fatal_error("UnicodePropertyKernelBuilder requires a non-boundary property expression");
    }
    Var * const property_stream = getOutputStreamVar("property_stream");
    pb.createAssign(pb.createExtract(property_stream, pb.getInteger(0)), propertyVar);
}

UnicodePropertyBasis::UnicodePropertyBasis(LLVMTypeSystemInterface & ts, UCD::EnumeratedPropertyObject * enumObj, StreamSet * Source, StreamSet * PropertyBasis)
: PabloKernel(ts,
              std::to_string(Source->getNumElements()) +
              "x" + std::to_string(Source->getFieldWidth()) +
              "UCD:" + getPropertyFullName(enumObj->getPropertyCode()) + "_basis",
{Binding{"source", Source}},
{Binding{"property_basis", PropertyBasis}})
, mEnumObj(enumObj) {

}

void UnicodePropertyBasis::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
    std::vector<UCD::UnicodeSet> & basisSets = mEnumObj->GetEnumerationBasisSets();
    std::vector<Var *> targetVars(basisSets.size());
    for (unsigned i = 0; i < basisSets.size(); i++) {
        std::string vname = "basis" + std::to_string(i);
        targetVars[i] = pb.createVar(vname, pb.createZeroes());
    }
    unicodeCompiler.compile(targetVars, basisSets);
    Var * const property_basis = getOutputStreamVar("property_basis");
    for (unsigned i = 0; i < targetVars.size(); i++) {
        pb.createAssign(pb.createExtract(property_basis, pb.getInteger(i)), targetVars[i]);
    }
}
