#
# nfc_generator.py (python3)
#
# Robert D. Cameron
# April 2025
#
# Licensed under Open Software License 3.0.
#
import re, string, os.path, cformat, codecs
import UCD_config
from unicode_set import *
from UCD_parser import *

import math

def ceil_log2(x):
    return math.ceil(math.log2(x))

class UCD_database():
    def __init__(self):
        self.supported_props = []
        self.property_data_headers = []
        self.missing_specs = {}
        self.binary_properties = {}
        self.load_property_name_info()
        self.load_property_value_info()
        parse_UnicodeData_txt(self.property_object_map)
        parse_property_data(self.property_object_map['dt'], 'extracted/DerivedDecompositionType.txt')
        parse_property_data(self.property_object_map['ccc'], 'extracted/DerivedCombiningClass.txt')
        parse_property_data(self.property_object_map['CE'], 'CompositionExclusions.txt')
        fold_data = parse_CaseFolding_txt(self.property_object_map)
        self.decomp_map = self.property_object_map['dm'].cp_value_map
        self.dt_map = self.property_object_map['dt'].value_map
        self.ccc_map = self.property_object_map['ccc'].value_map
        self.cf_map = self.property_object_map['cf'].cp_value_map
        self.scf_map = self.property_object_map['scf'].cp_value_map
        self.CE_map = self.property_object_map['CE'].value_map


    def load_property_name_info(self):
        (self.property_enum_name_list, self.property_object_map) = parse_PropertyAlias_txt()
        self.property_lookup_map = getPropertyLookupMap(self.property_object_map)
        self.full_name_map = {}
        for p in self.property_enum_name_list:
            self.full_name_map[p] = self.property_object_map[p].getPropertyFullName()

    def load_property_value_info(self):
        initializePropertyValues(self.property_object_map, self.property_lookup_map)


def u8_encoded_length(cp):
    if cp <= 0x7f: return 1
    if cp <= 0x7ff: return 2
    if cp <= 0xffff: return 3
    return 4

def u8_code_unit(cp, n):
    lgth = u8_encoded_length(cp)
    if n == 1:
        if lgth == 1: return cp
        if lgth == 2: return 0xC0 | (cp >> 6)
        if lgth == 3: return 0xE0 | (cp >> 12)
        if lgth == 4: return 0xF0 | (cp >> 18)
    else:
        return 0x80 | ((cp >> (6 * (lgth - n))) & 0x3F)

def uset_structural_key(uset):
    rgs = uset_to_range_list(uset)
    if len(rgs) == 0: return "empty"
    (lo, hi) = rgs[0]
    if lo <= 0x7F:
        s = "ASC"
    else:
        s = "%X" % u8_code_unit(lo, 1)
    last = 0
    for rg in rgs:
        (lo, hi) = rg
        if (lo & 0xFFFFF0) == (last & 0xFFFFF0):
            s += "_%x" % (lo & 0x0F)
        elif (lo & 0xFFFF00) == (last & 0xFFFF00):
            s += "_%x" % (lo & 0xFF)
        else:
            s += "_%x" % lo
        last = lo
        if lo < hi:
            if lo + 1 == hi:
                s += "_"
            else:
                s += "___"
            if (hi & 0xFFFFF0) == (last & 0xFFFFF0):
                s += "%x" % (hi & 0x0F)
            elif (hi & 0xFFFF00) == (last & 0xFFFF00):
                s += "%x" % (hi & 0xFF)
            else:
                s += "%x" % hi
            last = hi
    return s

scoped_compiler_decl_template = r"""    UTF::UTF_Compiler ${pfx}_compiler(getInputStreamVar("Basis"), ${scope}, pablo::BitMovementMode::${movement});
"""

scoped_compilation_template = r"""    std::vector<Var *> ${pfx}_vars(${num_roles});
    std::vector<UnicodeSet> ${pfx}_usets(${num_roles});
${assignments}
    ${pfx}_compiler.compile(${pfx}_vars, ${pfx}_usets);
"""

assignment_template = r"""    Var * ${role} = ${scope}.createVar("${role}", All0);
    ${pfx}_vars[${index}] = ${role};
    ${pfx}_usets[${index}] = ${role}_uset;
"""

external_template = r"""    Var * ${role} = ${scope}.createVar("${role}", All0);
    ${pfx}_vars[${index}] = ${role};
    ${pfx}_usets[${index}] = ${external_expr};
"""

class Uset_Builder:
    def __init__(self):
        self.installed_usets = {}
        self.external_usets = {}
        self.role_to_key_map = {}
        self.current_scope_roles = []

    def open_scope(self, pfx, scope):
        self.scope_pfx = pfx
        self.current_scope = scope
        self.current_scope_roles = []
        if pfx != "": return "    {\n"

    def install_external_uset(self, external_key, uset_expr):
        if not external_key in self.external_usets.keys():
            #print("new key: %s" % key)
            self.external_usets[external_key] = uset_expr
        if not external_key in self.current_scope_roles:
            self.current_scope_roles.append(external_key)
        return external_key

    def install_uset(self, uset, extra_lookahead = 0):
        key = uset_structural_key(uset)
        if extra_lookahead != 0:
            key += "_la_%i" % extra_lookahead
        if not key in self.installed_usets.keys():
            #print("new key: %s" % key)
            self.installed_usets[key] = uset
        if not key in self.current_scope_roles:
            self.current_scope_roles.append(key)
        return key

    def get_installed_usets(self):
        return self.installed_usets

    def generate_uset_definitions(self, name_template):
        defs = ""
        i = 0
        generated = {}
        for key in sorted(self.installed_usets.keys()):
            uset = self.installed_usets[key]
            set_name = name_template % i
            defs += "    " + uset.generate(set_name)
            generated[key] = set_name
            i += 1
            defs += "    const UnicodeSet & %s_uset = %s;\n" % (key, generated[key])
        return defs

    def get_uset_definitions(self):
        return self.generate_uset_definitions("uset_%i")

    def generate_scope_compilations(self, movement_mode = "LookAhead", extra_lookahead = 0):
        pfx = self.scope_pfx
        if movement_mode != "LookAhead":
            pfx = pfx + "_adv"
        elif extra_lookahead != 0:
            pfx = "%s_la_%i" % (pfx, extra_lookahead)
        t1 = string.Template(scoped_compiler_decl_template)
        defs = t1.substitute(pfx = pfx, 
                             scope = self.current_scope,
                             movement = movement_mode)
        t2 = string.Template(scoped_compilation_template)
        t3 = string.Template(assignment_template)
        t4 = string.Template(external_template)
        assigs = ""
        for i in range(len(self.current_scope_roles)):
            k = self.current_scope_roles[i]
            if k in self.installed_usets.keys():
                assigs += t3.substitute(pfx = pfx, scope = self.current_scope, index = i, role = k)
            else:
                expr = self.external_usets[k]
                assigs += t4.substitute(pfx = pfx, scope = self.current_scope, index = i, role = k, external_expr = expr)
        if extra_lookahead != 0:
            defs += "    %s_compiler.setExtraLookahead(%i);\n" % (pfx, extra_lookahead)
        defs += t2.substitute(pfx = pfx,
                              scope = self.current_scope,
                              num_roles = len(self.current_scope_roles),
                              assignments = assigs)
        self.current_scope_roles = []
        return defs

    def close_scope(self):
        return "    }\n"

def get_pfx_code(cp):
    initial = u8_code_unit(cp, 1)
    if initial <= 0x7F: return 0
    if initial <= 0xC3: return 0xC2
    if initial <= 0xDF:
        # strip low 2 bits
        return initial & 0xFC
    return initial

def pfx_code_string(pfx_code):
    if pfx_code == 0:
        return "x0_7F"
    if pfx_code <= 0xDF:
        return "x%x_%x" % (pfx_code, pfx_code | 0x03)
    return "x%x" % pfx_code

def pfx_code_lgth(pfx_code):
    if pfx_code == 0:
        return 1
    elif pfx_code <= 0xDF:
        return 2
    elif pfx_code <= 0xEF:
        return 3
    return 4

# TODO: consider possible optimization based on exact initial byte range
def prefix_test_logic(pfx_code):
    if pfx_code == 0:
        return "pb.createNot(Basis[7])"
    elif pfx_code <= 0xDF:
        return "pb.createAnd(bnc.UGE(Basis, 0x%x), bnc.ULE(Basis, 0x%x))" % (pfx_code, pfx_code | 3)
    else:
        return "bnc.EQ(Basis, 0x%x)" % pfx_code

def range_usets_from_cps(cp_list):
    rg_sets = {}
    for cp in cp_list:
        code = get_pfx_code(cp)
        if not code in rg_sets.keys():
            rg_sets[code] = singleton_uset(cp)
        else:
            rg_sets[code] = uset_union(rg_sets[code], singleton_uset(cp))
    return rg_sets

singleton_header = r"""//
SingletonCanonicalization::SingletonCanonicalization
    (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * XfrmBasis)
: PabloKernel(ts, "SingletonCanonicalization" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"XfrmBasis", XfrmBasis}}) {}

void SingletonCanonicalization::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    Var * DeleteVar = pb.createVar("DeleteVar", All0);
    std::vector<Var *> XfrmVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        XfrmVar[i] = pb.createVar("XfrmBasis" + std::to_string(i), All0);
    }
"""

nested_pfx_template = r"""
    auto ${builder} = pb.createScope();
    PabloAST * ${pfx_test_var} = ${logic};
    pb.createIf(${pfx_test_var}, ${builder});
    std::vector<PabloAST *> xfrm_${code_str}(8, All0);
    PabloAST * del_${code_str} = All0;
"""

def gen_nested_pfx_code(pfx_code):
    code_str = pfx_code_string(pfx_code)
    pfx_test_var = "pfx_%s_test" % code_str
    test = prefix_test_logic(pfx_code)
    scope = "b_%s" % code_str
    t = string.Template(nested_pfx_template)
    return t.substitute(builder = scope,
                        code_str = code_str,
                        pfx_test_var = pfx_test_var,
                        logic = test)

singleton_final_code = r"""
    Var * XfrmOutputVar = getOutputStreamVar("XfrmBasis");
    PabloAST * select = pb.createNot(DeleteVar);
    for (unsigned i = 0; i < 8; i++) {
        Var * xfrm_out = pb.createExtract(XfrmOutputVar, pb.getInteger(i));
        //  pb.createAssign(xfrm_out, XfrmVar[i]);
        pb.createAssign(xfrm_out, pb.createAnd(select, pb.createXor(Basis[i], XfrmVar[i])));
    }
}
"""

excluded_composite_header = r"""//
ExcludedCompositeStage::ExcludedCompositeStage
    (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * XfrmBasis)
: PabloKernel(ts, "ExcludedCompositeStage" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"XfrmBasis", XfrmBasis}}) {}

void ExcludedCompositeStage::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    Var * DeleteVar = pb.createVar("DeleteVar", All0);
    std::vector<Var *> XfrmVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        XfrmVar[i] = pb.createVar("XfrmBasis" + std::to_string(i), All0);
    }
"""

excluded_composite_final_code = r"""
    Var * XfrmOutputVar = getOutputStreamVar("XfrmBasis");
    PabloAST * select = pb.createNot(DeleteVar);
    for (unsigned i = 0; i < 8; i++) {
        Var * xfrm_out = pb.createExtract(XfrmOutputVar, pb.getInteger(i));
        //  pb.createAssign(xfrm_out, XfrmVar[i]);
        pb.createAssign(xfrm_out, pb.createAnd(select, pb.createXor(Basis[i], XfrmVar[i])));
    }
}
"""


pass_template = r"""//
class FindComposables${pass_no} : public PabloKernel {
public:
    FindComposables${pass_no}
        (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * ccc_NR,
                                       StreamSet * MarkCode, StreamSet * Index_ccc_NR_or_MarksFound);
protected:
    void generatePabloMethod() override;
};

FindComposables${pass_no}::FindComposables${pass_no}
    (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * ccc_NR,
                                   StreamSet * MarkCode, StreamSet * Index_ccc_NR_or_MarksFound)
: PabloKernel(ts, "FindComposables${pass_no}_" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}, Binding{"ccc_NR", ccc_NR}},
{Binding{"MarkCode", MarkCode}, Binding{"Index_ccc_NR_or_MarksFound", Index_ccc_NR_or_MarksFound}}) {}

void FindComposables${pass_no}::generatePabloMethod() {
    EnumeratedPropertyObject * cccObj = cast<EnumeratedPropertyObject>(getPropertyObject(ccc));
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    PabloAST * ccc_NR = getInputStreamSet("ccc_NR")[0];
    const unsigned markCodeBits = ${markCodeBits};
    std::vector<Var *> markCode(markCodeBits);
    for (unsigned i = 0; i < markCodeBits; i++) {
        markCode[i] = pb.createVar("markCode" + std::to_string(i), All0);
    }
"""

def gen_pass_header_code(pass_no, mark_code_bits):
    s = string.Template(pass_template)
    extra_inputs = ""
    extra_bindings = ""
    return s.substitute(pass_no = pass_no,
                        markCodeBits = mark_code_bits)

pass_pfx_template = r"""
    auto ${builder} = pb.createScope();
    PabloAST * ${pfx_test_var} = ${logic};
    pb.createIf(${pfx_test_var}, ${builder});
"""

def gen_pass_pfx_code(pfx_code):
    code_str = pfx_code_string(pfx_code)
    pfx_test_var = "pfx_%s_test" % code_str
    test = prefix_test_logic(pfx_code)
    scope = "b_%s" % code_str
    t = string.Template(pass_pfx_template)
    return t.substitute(builder = scope,
                        pfx_test_var = pfx_test_var,
                        logic = test)

mark_case_template = r"""
//  Case for mark ${mark}
    PabloAST * ${builder}_possible_${mark}_pos = ${builder}.createAdvanceThenScanTo(${mark_starters}, ${ccc_v}_or_NR);
    PabloAST * ${builder}_found_${mark} = ${builder}.createAnd(${builder}_possible_${mark}_pos, ${mark_var});
"""

mark_code_template = r"""    ${builder}.createAssign(markCode[${code_bit}], ${builder}.createOr(markCode[${code_bit}], ${builder}_found_${mark}));
"""

def gen_mark_case_logic(mark, starters_var, mark_var, ccc_enum, code_str, mark_code):
    t = string.Template(mark_case_template)
    s = t.substitute(builder = "b_%s" % code_str,
                        mark="%x" % mark,
                        mark_var = mark_var,
                        mark_starters=starters_var,
                        ccc_v= "ccc_%s_%s" % (ccc_enum, code_str))
    t2 = string.Template(mark_code_template)
    i = 0
    while mark_code > 0:
        if (mark_code & 1) == 1:
            s += t2.substitute(builder = "b_%s" % code_str, mark="%x" % mark, code_bit = "%i" % i)
        i += 1
        mark_code = mark_code >> 1
    return s

finalize_nfc_template = r"""// Generate combined outputs for pass ${pass_no}.
    Var * markOutputVar = getOutputStreamVar("MarkCode");
    for (unsigned i = 0; i < markCodeBits; i++) {
        pb.createAssign(pb.createExtract(markOutputVar, pb.getInteger(i)), markCode[i]);
    }
    PabloAST * updatedIndexStrm = pb.createOr(ccc_NR, composable2nd);
    Var * indexVar = getOutputStreamVar("Index_ccc_NR_or_MarksFound");
    pb.createAssign(pb.createExtract(indexVar, pb.getInteger(0)), updatedIndexStrm);
}
"""

def finalize_nfc_stage(pass_no):
    s = string.Template(finalize_nfc_template)
    return s.substitute(pass_no = pass_no)

long_composable_application_template = r"""//
class ApplyLongComposition${pass_no} : public PabloKernel {
public:
    ApplyLongComposition${pass_no}
        (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * MarkCodeAtStarter, StreamSet * MarkCode,
                                       StreamSet * OutputBasis);
protected:
    void generatePabloMethod() override;
};

ApplyLongComposition${pass_no}::ApplyLongComposition${pass_no}
    (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * MarkCodeAtStarter, StreamSet * MarkCode,
                                   StreamSet * OutputBasis)
: PabloKernel(ts, "ApplyLongComposition${pass_no}_" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}, Binding{"MarkCodeAtStarter", MarkCodeAtStarter}, Binding{"MarkCode", MarkCode}},
{Binding{"OutputBasis", OutputBasis}}) {}

void ApplyLongComposition${pass_no}::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    std::vector<PabloAST *> markCodeAtStarter = getInputStreamSet("MarkCodeAtStarter");
    const unsigned markCodeBits = ${markCodeBits};
    PabloAST * markFoundForStarter = markCodeAtStarter[0];
    for (unsigned i = 1; i < markCodeBits; i++) {
        markFoundForStarter = pb.createOr(markFoundForStarter, markCodeAtStarter[i]);
    }
    std::vector<PabloAST *> markCode = getInputStreamSet("MarkCode");
    PabloAST * anyMark = markCode[0];
    for (unsigned i = 1; i < markCodeBits; i++) {
        anyMark = pb.createOr(anyMark, markCode[i]);
    }
    std::vector<Var *> XfrmVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        XfrmVar[i] = pb.createVar("XfrmBasis" + std::to_string(i), All0);
    }
"""

def gen_long_composition_xfrm_header_code(pass_no, mark_code_bits):
    s = string.Template(long_composable_application_template)
    return s.substitute(pass_no = pass_no,
                        markCodeBits = mark_code_bits)

long_composition_pfx_template = r"""
    auto ${builder} = pb.createScope();
    PabloAST * ${pfx_test_var} = ${logic};
    pb.createIf(pb.createAnd(${pfx_test_var}, markFoundForStarter), ${builder});
    std::vector<PabloAST *> xfrm_${code_str}(8, All0);
    BixNumCompiler ${builder}_bnc(${builder});
"""

def gen_long_composition_pfx_code(pfx_code):
    code_str = pfx_code_string(pfx_code)
    pfx_test_var = "pfx_%s_test" % code_str
    test = prefix_test_logic(pfx_code)
    scope = "b_%s" % code_str
    t = string.Template(long_composition_pfx_template)
    return t.substitute(builder = scope,
                        code_str = code_str,
                        pfx_test_var = pfx_test_var,
                        logic = test)

long_composable_final_code = r"""
    anyMark = pb.createOr(pb.createAdvance(pb.createAnd(bnc.UGE(Basis, 0xC2), anyMark), 1), anyMark);
    anyMark = pb.createOr(pb.createAdvance(pb.createAnd(bnc.UGE(Basis, 0xE0), anyMark), 2), anyMark);
    anyMark = pb.createOr(pb.createAdvance(pb.createAnd(bnc.UGE(Basis, 0xF0), anyMark), 3), anyMark);
    PabloAST * selectMask = pb.createNot(anyMark);
    Var * XfrmOutputVar = getOutputStreamVar("OutputBasis");
    for (unsigned i = 0; i < 8; i++) {
        Var * xfrm_out = pb.createExtract(XfrmOutputVar, pb.getInteger(i));
        //  pb.createAssign(xfrm_out, XfrmVar[i]);
        pb.createAssign(xfrm_out, pb.createAnd(selectMask, pb.createXor(Basis[i], XfrmVar[i])));
    }
}
"""

long_composable_pipeline_step_template = r"""//  Pass ${pass_no} to identify long composable sequences and transform to precomposed characters.
    StreamSet * MarkCode${step} = P.CreateStreamSet(${mark_code_bits});
    StreamSet * Index_ccc_NR_or_MarksFound${step} = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<FindComposables${pass_no}>(${input_basis}, ccc_NR, MarkCode${step}, Index_ccc_NR_or_MarksFound${step});
    SHOW_BIXNUM(MarkCode${step});
    SHOW_STREAM(Index_ccc_NR_or_MarksFound${step});

    StreamSet * MarkCodeAtStarter${step} = P.CreateStreamSet(${mark_code_bits}, 1);
    P.CreateKernelCall<IndexedShiftBack>(Index_ccc_NR_or_MarksFound${step}, MarkCode${step}, MarkCodeAtStarter${step});
    SHOW_BIXNUM(MarkCodeAtStarter${step});

    P.CreateKernelCall<ApplyLongComposition${pass_no}>(${input_basis}, MarkCodeAtStarter${step},  MarkCode${step}, ${output_basis});
    SHOW_BIXNUM(${output_basis});
"""

long_composable_pipeline_template = r"""void LongComposablePipeline(PipelineBuilder & P,
                            StreamSet * Basis, StreamSet * ccc_NR, StreamSet * FinalBasis) {
${pipeline_logic}
}
"""

short_composable_header = r"""//
ShortComposableTranslation::ShortComposableTranslation
    (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * OutputBasis)
: PabloKernel(ts, "ShortComposableTranslation" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(7)}},
{Binding{"OutputBasis", OutputBasis}}) {}

void ShortComposableTranslation::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    Var * DeleteVar = pb.createVar("DeleteVar", All0);
    std::vector<Var *> XfrmVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        XfrmVar[i] = pb.createVar("XfrmVar" + std::to_string(i), All0);
    }
"""

short_composable_case_template = r"""//     ${cp1} + ${cp2} => ${precomposed}
    PabloAST * found_${cp1}_before_${cp2} = b_${code_str}.createAnd(${cp1_var}, ${cp2_var});
    del_${code_str} = b_${code_str}.createOr(del_${code_str}, found_${cp1}_before_${cp2});
    PabloAST * found_${cp1}_${cp2} = b_${code_str}.createAdvance(found_${cp1}_before_${cp2}, ${cp1_len});
"""

def generate_short_case_code(builder, code_str, cp1, cp2, generated, generated_seconds, precomposed):
    t = string.Template(short_composable_case_template)
    s = t.substitute(code_str = code_str,
                        cp1 = "%x" % cp1,
                        cp2 = "%x" % cp2,
                        cp1_var = generated[cp1],
                        cp2_var = generated_seconds[cp2],
                        cp1_len=u8_encoded_length(cp1),
                        precomposed = "%x" % precomposed)
    to_xlate = "found_%x_%x" % (cp1, cp2)
    xlate_map = {cp2 : chr(precomposed)}
    bit_xfrm_sets = u8_bit_transform_sets(xlate_map)
    del_usets = u8_deletion_sets(xlate_map)
    bit_xfrm_data = install_bit_xfrm_usets(builder, bit_xfrm_sets)
    del_vars = install_del_usets(builder, del_usets)
    #xlate_code = generateUpdateBitXfrms("b_%s" % code_str, bit_xfrm_data, "xfrm_%s" % code_str, to_xlate)
    xlate_code = CharacterTranslationLogic(cp2, precomposed, to_xlate, "xfrm_%s" % code_str, "b_%s" % code_str)
    return s + xlate_code

short_composable_pfx_template = r"""
    auto ${builder} = pb.createScope();
    PabloAST * ${pfx_test_var} = ${logic};
    pb.createIf(${pfx_test_var}, ${builder});
    std::vector<PabloAST *> xfrm_${code_str}(8, All0);
    PabloAST * del_${code_str} = All0;
"""

def gen_short_composable_pfx_code(pfx_code):
    code_str = pfx_code_string(pfx_code)
    pfx_test_var = "pfx_%s_test" % code_str
    test = prefix_test_logic(pfx_code)
    scope = "b_%s" % code_str
    t = string.Template(short_composable_pfx_template)
    return t.substitute(builder = scope,
                        code_str = code_str,
                        pfx_test_var = pfx_test_var,
                        logic = test)

short_composable_pfx_final_template = r"""
    for (unsigned i = 0; i < 8; i++) {
        ${builder}.createAssign(XfrmVar[i], $builder.createOr(XfrmVar[i], xfrm_${code_str}[i]));
    }
    ${builder}.createAssign(DeleteVar, $builder.createOr(DeleteVar, del_${code_str}));
"""

def finalize_short_composable_pfx_code(pfx_code):
    code_str = pfx_code_string(pfx_code)
    scope = "b_%s" % code_str
    t = string.Template(short_composable_pfx_final_template)
    return t.substitute(builder = scope,
                        code_str = code_str,
                        cp1_len = pfx_code_lgth(pfx_code))

short_composable_final_code = r"""
    PabloAST * delspan = pb.createOr(DeleteVar, pb.createAdvance(pb.createAnd(bnc.UGE(Basis, 0xC2), DeleteVar), 1));
    delspan = pb.createOr(delspan, pb.createAdvance(pb.createAnd(bnc.UGE(Basis, 0xE0), DeleteVar), 2));
    delspan = pb.createOr(delspan, pb.createAdvance(pb.createAnd(bnc.UGE(Basis, 0xF0), DeleteVar), 3));
    PabloAST * select = pb.createNot(delspan);
    Var * OutputVar = getOutputStreamVar("OutputBasis");
    for (unsigned i = 0; i < 8; i++) {
        Var * vi = pb.createExtract(OutputVar, pb.getInteger(i));
        pb.createAssign(vi, pb.createAnd(select, pb.createXor(XfrmVar[i], Basis[i])));
    }
}
"""

self_composable_CC_header = r"""//
SelfComposableCCs::SelfComposableCCs
    (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * self_composable_CCs)
: PabloKernel(ts, "SelfComposableCCs" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"self_composable_CCs", self_composable_CCs}}) {}

void SelfComposableCCs::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
"""

self_composable_CC_final_code = r"""
    writeOutputStreamSet("self_composable_CCs", self_composables);
}
"""

self_composable_header = r"""//
SelfComposableTranslation::SelfComposableTranslation
    (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * self_composable_CCs,
                                   StreamSet * XfrmedBasis)
: PabloKernel(ts, "SelfComposableTranslation" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}, Binding{"self_composable_CCs", self_composable_CCs, FixedRate(), LookAhead(4)}},
{Binding{"XfrmedBasis", XfrmedBasis}}) {}

void SelfComposableTranslation::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    std::vector<PabloAST *> self_composable_CCs = getInputStreamSet("self_composable_CCs");
    Var * DelVar = pb.createVar("DelVar", All0);
    std::vector<Var *> XfrmVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        XfrmVar[i] = pb.createVar("XfrmBasis" + std::to_string(i), All0);
    }
"""

self_composable_case_template = r"""    auto b_${A} = pb.createScope();
    pb.createIf(pb.createOr(${A_AST}, ${AA_AST}), b_${A});
    std::vector<PabloAST *> basisXor_${A}(8, All0);
    SCResults rslt_${A} = SelfComposableLogic(b_${A}, Basis, ${A_len}, ${AA_len}, ${A_AST}, ${AA_AST});
    b_${A}.createAssign(DelVar, b_${A}.createOr(DelVar, rslt_${A}.A_to_delete));
    PabloAST * xfrm_${A} = b_${A}.createOr(rslt_${A}.A_to_convert_to_AA, rslt_${A}.AA_to_convert_to_A);
"""

self_composable_final_code = r"""
    PabloAST * select = pb.createNot(DelVar);
    Var * XfrmOutputVar = getOutputStreamVar("XfrmedBasis");
    for (unsigned i = 0; i < 8; i++) {
        Var * xfrm_out = pb.createExtract(XfrmOutputVar, pb.getInteger(i));
        pb.createAssign(xfrm_out, pb.createAnd(select, pb.createXor(XfrmVar[i], Basis[i])));
    }
}
"""

def u8_byte_xfrms(cp, strg):
    len1 = u8_encoded_length(cp)
    len2 = 0
    for ch in strg:
        len2 += u8_encoded_length(ord(ch))
    strg_bytes = []
    for ch in strg:
        cp2 = ord(ch)
        for i in range(u8_encoded_length(cp2)):
            strg_bytes.append(u8_code_unit(cp2, i + 1))
    if len1 > len2:
        strg_bytes += [0 for i in range(len1 - len2)]
    xfrm_bytes = []
    for i in range(len(strg_bytes)):
        cp_byte = 0 # default for positions corresponding to inserted zeroes
        if i < len1:
            cp_byte = u8_code_unit(cp, i + 1)
        xfrm_bytes.append(cp_byte ^ strg_bytes[i])
    return xfrm_bytes
#
#  UTF-8 character translation using Xor method for cp1 -> cp2.
#  Assumptions:
#    - marker is a bit stream marking the first byte of any cp1
#    - if u8_encoded_length(cp2) > u8_encoded_length(cp1):
#      zeroes have been inserted after the last byte of cp1
#    - if u8_encoded_length(cp2) < u8_encoded_length(cp1):
#      excess = u8_encoded_length(cp1) - u8_encoded_length(cp2)
#    - pb is the PabloBuilder for logic
#
def CharacterTranslationLogic(cp1, cp2, marker, basis_var, pb):
    xfrm_bytes = u8_byte_xfrms(cp1, chr(cp2))
    s = ""
    len2 = u8_encoded_length(cp2)
    for i in range(len2):
        diff = xfrm_bytes[i]
        if diff != 0:
            # advance marker to pos
            if i == 0:
                s += "    PabloAST * m_%x_%x_0 = %s;\n" % (cp1, cp2, marker)
            else:
                s += "    PabloAST * m_%x_%x_%i = %s.createAdvance(%s, %i);\n" % (cp1, cp2, i, pb, marker, i)
            # apply xor logic for each bit difference between cp1_byte, cp2_byte
            for j in range(8):
                bv = "%s[%i]" % (basis_var, j)
                if ((diff >> j) & 1) == 1:
                    s += "    %s = %s.createOr(%s, m_%x_%x_%i);\n" % (bv, pb, bv, cp1, cp2, i)
    return s

def u8_deletion_sets(char2string_map):
    deletion_usets = {}
    for cp in char2string_map.keys():
        cp_uset = singleton_uset(cp)
        len1 = u8_encoded_length(cp)
        str = char2string_map[cp]
        str_bytes = []
        for ch in str:
            cp1 = ord(ch)
            for i in range(u8_encoded_length(cp1)):
                str_bytes.append(u8_code_unit(cp1, i + 1))
        len2 = len(str_bytes)
        if len1 > len2:
            ldiff = len1 - len2
            #print("cp %x, ldiff: %i" % (cp, ldiff))
            if not ldiff in deletion_usets.keys():
                deletion_usets[ldiff] = empty_uset()
            deletion_usets[ldiff] = uset_union(deletion_usets[ldiff], cp_uset)
        #if len(deletion_usets.keys()) > 0:
            #for diff in deletion_usets.keys():
                #print("deletion_uset[%i]:" % diff)
    return deletion_usets

def u8_bit_transform_sets(char2string_map):
    bit_xfrm_sets = {}
    for cp in char2string_map.keys():
        cp_uset = singleton_uset(cp)
        xfrm_bytes = u8_byte_xfrms(cp, char2string_map[cp])
        for i in range(len(xfrm_bytes)):
            diff = xfrm_bytes[i]
            if diff != 0:
                if not i in bit_xfrm_sets.keys():
                    bit_xfrm_sets[i] = {}
                for j in range(8):
                    if (diff >> j) & 1 == 1:
                        if not j in bit_xfrm_sets[i].keys():
                            bit_xfrm_sets[i][j] = empty_uset()
                        bit_xfrm_sets[i][j] = uset_union(bit_xfrm_sets[i][j], cp_uset)
    return bit_xfrm_sets

def install_del_usets(builder, del_sets):
    del_vars = {}
    for del_amt in del_sets.keys():
        del_vars[del_amt] = builder.install_uset(del_sets[del_amt])
    return del_vars

def install_bit_xfrm_usets(builder, bit_xfrm_sets):
    bit_xfrm_data = {}
    for pos in sorted(bit_xfrm_sets.keys()):
        for bit in bit_xfrm_sets[pos].keys():
            uset_key = builder.install_uset(bit_xfrm_sets[pos][bit])
            if not uset_key in bit_xfrm_data.keys():
                bit_xfrm_data[uset_key] = []
            bit_xfrm_data[uset_key].append((pos, bit))
    return bit_xfrm_data

def generateUpdateBitXfrms(scope, bit_xfrm_data, basis, marker):
    usets = sorted(bit_xfrm_data.keys())
    s = "    std::vector<PabloAST *> usets(%i);\n" % len(usets)
    spec_len = 0
    for i in range(len(usets)):
        s += "    usets[%i] = %s;\n" % (i, usets[i])
        spec_len += len(bit_xfrm_data[usets[i]])
    s += "    std::vector<BitXfrmSpec> xfrmSpecs(%i);\n" % spec_len
    j = 0
    for i in range(len(usets)):
        for (pos, bit) in bit_xfrm_data[usets[i]]:
            s += "    xfrmSpecs[%i] = {%i, %i, %i};\n" % (j, i, pos, bit)
            j += 1
    s += "    UpdateBitXfrms(%s, %s, %s, usets, xfrmSpecs);\n" % (scope, basis, marker)
    return s

u8_insertion_bixnum_template = r"""//
NFC_Initial_Insertion::NFC_Initial_Insertion
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * InsertionBixNum)
: PabloKernel(ts, "NFC_Initial_Insertion" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"InsertionBixNum", InsertionBixNum}}) {}

void NFC_Initial_Insertion::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    std::vector<PabloAST *> insertions(${insertion_bixnum_bits}, All0); 
"""

u8_insertion_final_code = r"""    writeOutputStreamSet("InsertionBixNum", insertions);
}
"""

def insert_map_to_bixnum_usets(ins_map):
    bixnum_usets = []
    for cp in ins_map.keys():
        num = ins_map[cp]
        uset1 = singleton_uset(cp)
        i = 0
        bit = 1 << i
        while bit <= num:
            if bit & num == bit:  # bit is set
                while len(bixnum_usets) <= i:
                    bixnum_usets.append(empty_uset())
                bixnum_usets[i] = uset_union(bixnum_usets[i], uset1)
            i += 1
            bit = 1 << i
    return bixnum_usets

candidate_class_template = r"""//  The NFC_CandidateClass kernel produces the class of characters 
//  that are relevant to NFC processing by virtue of being reorderable marks or
//  non-reorderable characters that can occur as the second character of a
//  composable sequence.
//
//class NFC_CandidateClass : public pablo::PabloKernel {
//public:
//NFC_CandidateClass
//    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
//                                   StreamSet * NFC_CandidateClass);
//protected:
//    void generatePabloMethod() override;
//};

NFC_CandidateClass::NFC_CandidateClass
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * candidates)
: PabloKernel(ts, "NFC_CandidateClass" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"candidates", candidates}}) {}

void NFC_CandidateClass::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    Var * Basis = getInputStreamVar("Basis");
    UTF::UTF_Compiler utf_compiler(Basis, pb, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> targets = {pb.createVar("candidates", pb.createZeroes()), 
                                  pb.createVar("expandfirst", pb.createZeroes())};
    utf_compiler.compile(targets, {${candidate_uset}_uset, ${expansion_required_uset}_uset});
    writeOutputStreamSet("candidates", targets);
}
"""

nfc_generated_cpp_template = r"""#include <kernel/unicode/normalization.h>
#include <unicode/core/unicode_set.h>
#include <unicode/data/PropertyObjectTable.h>
#include <unicode/algo/normalization.h>
#include <unicode/utf/utf_compiler.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
#include <kernel/streamutils/stream_shift.h>
#include <toolchain/toolchain.h>
#include <kernel/pipeline/pipeline_builder.h>

using namespace pablo;
using namespace kernel;
using namespace llvm;
using namespace UCD;

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

${uset_declarations}

${kernels}
"""


class NFC_generator:
    def __init__(self, ucd):
        super().__init__()
        self.ucd = ucd
        self.builder = Uset_Builder()
        self.ccc_val_map = ucd.property_object_map['ccc'].cp_value_map
        self.ccc_enum_map = ucd.property_object_map['ccc'].property_value_enum_integer
        self.ccc_enum_rmap = ucd.property_object_map['ccc'].enum_integer_to_value_map
        self.pass_count = 0
        self.ccc_pass_allocation = {}
        self.pass_cccs = []
        self.initialize_prefix_based_ranges()

    def initialize_prefix_based_ranges(self):
        # ASCII range
        lo_cp = 0
        hi_cp = 0x7F
        self.prefix_based_ranges = [(lo_cp, hi_cp)]
        #
        #  Ranges based on groups of UTF-8 prefixes of length 2
        for i in range(8):
            lo_cp = hi_cp + 1
            hi_cp = lo_cp | 0xFF
            self.prefix_based_ranges.append((lo_cp, hi_cp))
        #  Ranges based on individual prefixes of length 3 (E0 ... EF)
        for i in range(16):
            lo_cp = hi_cp + 1
            hi_cp = lo_cp | 0xFFF
            self.prefix_based_ranges.append((lo_cp, hi_cp))
        #  Ranges based on individual prefixes of length 4 (F0 .. F4)
        for i in range(4):
            lo_cp = hi_cp + 1
            hi_cp = lo_cp | 0x3FFFF
            self.prefix_based_ranges.append((lo_cp, hi_cp))
        lo_cp = hi_cp + 1

    def show_prefix_based_ranges(self):
        self.prefix_based_ranges.append((lo_cp, 0x10FFFF))
        for rg in self.prefix_based_ranges:
            (rlo, rhi) = rg
            ulo = u8_code_unit(rlo, 1)
            uhi = u8_code_unit(rhi, 1)
            if ulo != uhi:
                print("Range %x-%x, initial_code_units %x-%x" % (rlo, rhi, ulo, uhi))
            else:
                print("Range %x-%x, initial_code_unit %x" % (rlo, rhi, ulo))

    def excluded_composite(self, cp):
        if not uset_member(ucd.dt_map['Can'], cp): return False
        decomp = ucd.decomp_map[cp]
        if len(decomp) == 1: return False
        if uset_member(ucd.CE_map['Y'], cp): return True
        if not uset_member(ucd.ccc_map['NR'], cp): return True
        return not uset_member(ucd.ccc_map['NR'], ord(decomp[0]))

    def full_decomp(self, cp):
        if not uset_member(ucd.dt_map['Can'], cp): return chr(cp)
        decomp = ucd.decomp_map[cp]
        cp1 = ord(decomp[0])
        while cp1 in ucd.decomp_map.keys() and uset_member(ucd.dt_map['Can'], cp1):
            decomp = ucd.decomp_map[cp1] + decomp[1:]
            cp1 = ord(decomp[0])
        return decomp

    def create_mappings(self):
        self.full_decomp_map = {}
        self.singleton_map = {}
        self.short_composable_map = {}
        self.composable_seconds = empty_uset()
        self.self_composables = {}
        self.long_composable_map = {}
        self.excluded_composite_map = {}
        self.non_starter_uset = empty_uset()
        for precomposed in ucd.decomp_map.keys():
            if uset_member(ucd.dt_map['Can'], precomposed):
                decomp = ucd.decomp_map[precomposed]
                cp1 = ord(decomp[0])
                self.full_decomp_map[precomposed] = self.full_decomp(precomposed)
                if len(decomp) == 1:
                    self.singleton_map[precomposed] = cp1
                elif len(decomp) == 2:
                    if not uset_member(ucd.ccc_map['NR'], precomposed) or not uset_member(ucd.ccc_map['NR'], cp1):
                        self.non_starter_uset = uset_union(self.non_starter_uset, singleton_uset(precomposed))
                    cp2 = ord(decomp[1])
                    if self.excluded_composite(precomposed):
                        while self.excluded_composite(cp1):
                            #print("Recursive excluded composite: %x => %x %x => %x %x %x" %(precomposed, cp1, cp2, ord(ucd.decomp_map[cp1][0]), ord(ucd.decomp_map[cp1][1]), cp2))
                            decomp = ucd.decomp_map[cp1] + decomp[1:]
                            cp1 = ord(decomp[0])
                        self.excluded_composite_map[precomposed] = decomp
                    else:
                        if uset_member(ucd.ccc_map['NR'], cp2):
                            # Decomposition to two consecutive starters
                            self.composable_seconds = uset_union(self.composable_seconds, singleton_uset(cp2))
                            if cp1 == cp2:
                                self.self_composables[cp1] = precomposed
                            else:
                                if not cp1 in self.short_composable_map.keys():
                                    # index by the first character of decomposition
                                    self.short_composable_map[cp1] = {}
                                self.short_composable_map[cp1][cp2] = precomposed
                        else:
                            if not cp2 in self.long_composable_map.keys():
                                # index by the mark (second char of decomposition)
                                self.long_composable_map[cp2] = {}
                            self.long_composable_map[cp2][cp1] = chr(precomposed)
                else:
                    raise Exception("Unexpected: decomposition length(%x) = %i" % (precomposed, len(decomp)))
        for cp in self.singleton_map.keys():
            canon_cp = self.singleton_map[cp]
            if self.excluded_composite(canon_cp):
                decomp = self.excluded_composite_map[canon_cp]
                self.excluded_composite_map[cp] = decomp
                print("singleton %x maps to excluded_composite %x => %s" % (cp, canon_cp, " ".join(["%x" % ord(c) for c in decomp])))
            if uset_member(ucd.dt_map['Can'], canon_cp):
                decomp = ucd.decomp_map[canon_cp]
                print("singleton %x maps to %x => %s" % (cp, canon_cp, " ".join(["%x" % ord(c) for c in decomp])))
        for cp1 in self.short_composable_map.keys():
            if uset_member(self.composable_seconds, cp1):
                for cp2 in self.short_composable_map[cp1].keys():
                    precomp = self.short_composable_map[cp1][cp2]
                    self.composable_seconds = uset_union(self.composable_seconds, singleton_uset(precomp))

    def add_doubleton_shorts(self):
        for A in sorted(self.self_composables.keys()):
            AA = self.self_composables[A]
            for x in self.short_composable_map.keys():
                if A in self.short_composable_map[x].keys():
                    y = self.short_composable_map[x][A]
                    if y in self.short_composable_map.keys():
                        if A in self.short_composable_map[y].keys():
                            z = self.short_composable_map[y][A]
                            self.short_composable_map[x][AA] = z

    def add_overridable_seconds(self):
        by_second = {}
        for cp1 in self.short_composable_map.keys():
            for cp2 in self.short_composable_map[cp1].keys():
                if not cp2 in by_second.keys():
                    by_second[cp2] = {}
                by_second[cp2][cp1] = self.short_composable_map[cp1][cp2]

        # look for overridable seconds: A BC ==> AB C
        for B in self.short_composable_map.keys():
            for C in self.short_composable_map[B].keys():
                BC = self.short_composable_map[B][C]
                if B in by_second.keys():
                    for A in by_second[B].keys():
                        # Now have an AB precomposed combo with a following C.
                        AB = by_second[B][A]
                        if AB in self.short_composable_map.keys():
                            if C in self.short_composable_map[AB].keys():
                                ABC = self.short_composable_map[AB][C]
                                self.short_composable_map[A][BC] = ABC
                                print("%x %x ==> %x %x ==> %x" %(A, BC, AB, C, ABC))
                        else:
                            print("%x %x ==> %x %x" %(A, BC, AB, C))
        for A in self.self_composables.keys():
            AA = self.self_composables[A]
            if A in self.short_composable_map.keys():
                if AA in self.short_composable_map.keys():
                    for X in self.short_composable_map[AA].keys():
                        AAX = self.short_composable_map[AA][X]
                        if X in self.short_composable_map[A].keys():
                            AX = self.short_composable_map[A][X]
                            self.short_composable_map[A][AX] = AAX
                            print("%x %x ==> %x %x ==> %x" %(A, AX, AA, X, AAX))
                for X in self.short_composable_map[A].keys():
                    if AA in self.short_composable_map.keys() and X in self.short_composable_map[AA].keys():
                        continue
                    AX = self.short_composable_map[A][X]
                    print("%x %x ==> %x %x" %(A, AX, AA, X))

    def cp_with_ccc(self, cp):
        return "%x(%s)" % (cp, self.ccc_val_map[cp])

    def display_singletons(self):
        print("Singletons:")
        for cp in sorted(self.singleton_map.keys()):
            canon_cp = self.singleton_map[cp]
            print("%s => %s" % (self.cp_with_ccc(cp), self.cp_with_ccc(canon_cp)))

    def display_excluded_composites(self):
        print("Excluded Composites:")
        for cp in sorted(self.excluded_composite_map.keys()):
            decomp = self.excluded_composite_map[cp]
            print("%s => %s" % (self.cp_with_ccc(cp), " ".join([self.cp_with_ccc(ord(c)) for c in decomp])))

    def display_short_composables(self):
        print("Short Composables:")
        for cp1 in sorted(self.short_composable_map.keys()):
            print("%s: " % self.cp_with_ccc(cp1))
            for cp2 in sorted(self.short_composable_map[cp1].keys()):
                cp = self.short_composable_map[cp1][cp2]
                print("    + %s => %s" % (self.cp_with_ccc(cp2), self.cp_with_ccc(cp)))

    def display_self_composables(self):
        print("Self Composables:")
        for cp1 in sorted(self.self_composables.keys()):
            print("%x + %x => %x: " % (cp1, cp1, self.self_composables[cp1]))

    def display_long_composables(self):
        print("Long Composables:")
        by_starter = {}
        for cp2 in sorted(self.long_composable_map.keys()):
            for cp1 in self.long_composable_map[cp2].keys():
                s = self.long_composable_map[cp2][cp1]
                if not cp1 in by_starter.keys():
                    by_starter[cp1] = {}
                by_starter[cp1][cp2] = s
        for cp1 in sorted(by_starter.keys()):
            print("%s: " % self.cp_with_ccc(cp1))
            for cp2 in sorted(by_starter[cp1].keys()):
                s = by_starter[cp1][cp2]
                print("    + %s => %s" % (self.cp_with_ccc(cp2), " ".join(["%x" % ord(c) for c in s])))

    def display_recursive_decomposables(self):
        print("Recursive Composables:")
        for precomposed in sorted(ucd.decomp_map.keys()):
            if uset_member(ucd.dt_map['Can'], precomposed):
                decomp = ucd.decomp_map[precomposed]
                if ord(decomp[0]) in ucd.decomp_map.keys():
                    if uset_member(ucd.dt_map['Can'], ord(decomp[0])):
                        decomp_str1 = " ".join(["%x" % ord(c) for c in decomp])
                        decomp2 = ucd.decomp_map[ord(decomp[0])] + decomp[1:]
                        decomp_str2 = " ".join(["%x" % ord(c) for c in decomp2])
                        print ("%x => %s => %s" % (precomposed, decomp_str1, decomp_str2))

    def update_max_insert_map(self, cp, insert_len):
        if cp in self.max_insert_map.keys():
            if insert_len > self.max_insert_map[cp]:
                self.max_insert_map[cp] = insert_len
        elif insert_len > 0:
            self.max_insert_map[cp] = insert_len

    def create_max_insert_map(self):
        self.max_insert_map = {}
        for cp in self.singleton_map.keys():
            canon = self.singleton_map[cp]
            ldiff = u8_encoded_length(canon) - u8_encoded_length(cp)
            if ldiff > 0:
                self.update_max_insert_map(cp, ldiff)
        for cp in self.short_composable_map.keys():
            for cp2 in self.short_composable_map[cp].keys():
                precomposed = self.short_composable_map[cp][cp2]
                ldiff = u8_encoded_length(precomposed) - u8_encoded_length(cp)
                if ldiff > 0:
                    self.update_max_insert_map(cp, ldiff)
        for mark in self.long_composable_map.keys():
            for starter in self.long_composable_map[mark].keys():
                precomposed = self.long_composable_map[mark][starter]
                u8len = 0
                for c in precomposed:
                    u8len += u8_encoded_length(ord(c))
                ldiff = u8len - u8_encoded_length(starter)
                if ldiff > 0:
                    self.update_max_insert_map(starter, ldiff)
        for cp in self.excluded_composite_map.keys():
            decomp = self.excluded_composite_map[cp]
            decomp_len = 0
            for c in decomp:
                decomp_len += u8_encoded_length(ord(c))
            ldiff = decomp_len - u8_encoded_length(cp)
            if ldiff > 0:
                self.update_max_insert_map(cp, ldiff)
        for overridden in sorted(self.overridable_primaries.keys()):
            for mark in self.overridable_primaries[overridden].keys():
                decomp = self.overridable_primaries[overridden][mark]
                decomp_len = 0
                for c in decomp:
                    decomp_len += u8_encoded_length(ord(c))
                ldiff = decomp_len - u8_encoded_length(overridden)
                if ldiff > 0:
                    self.update_max_insert_map(overridden, ldiff)

    def display_max_insert_map(self):
        for cp in sorted(self.max_insert_map.keys()):
            print("%04x - insert %i" % (cp, self.max_insert_map[cp]))

    # Overridable primaries are primary composites p with
    # decomposition into starter s and mark m such that there
    # is a composite p' = <s, m'> where CCC(m) > CCC(m')
    # In this case the sequence <p ... m'> may be replaced by <p' m ...>
    #
    # A special case is when a second composite p'' = <p' m> exists,
    # in which case, the sequence <p ... m'> can directly be replaced
    # by <p'' ...>.
    #
    def determine_overridable_primaries(self):
        by_starter_and_ccc = {}
        for mark in self.long_composable_map.keys():
            ccc = self.ccc_val_map[mark]
            ccc_code = self.ccc_enum_map[ccc]
            for starter in self.long_composable_map[mark].keys():
                if not starter in by_starter_and_ccc.keys():
                    by_starter_and_ccc[starter] = {}
                if not ccc_code in by_starter_and_ccc[starter].keys():
                    by_starter_and_ccc[starter][ccc_code] = {}
                by_starter_and_ccc[starter][ccc_code][mark] = ord(self.long_composable_map[mark][starter])
        self.overridable_primaries = {}
        for starter in by_starter_and_ccc.keys():
            ccc_list = sorted(by_starter_and_ccc[starter].keys())
            for i in range(len(ccc_list) - 1):
                for j in range(i+1, len(ccc_list)):
                    lo_ccc = ccc_list[i]
                    hi_ccc = ccc_list[j]
                    for m1 in by_starter_and_ccc[starter][hi_ccc].keys():
                        overridable = by_starter_and_ccc[starter][hi_ccc][m1]
                        if not overridable in self.overridable_primaries.keys():
                            self.overridable_primaries[overridable] = {}
                        overrider_marks = by_starter_and_ccc[starter][lo_ccc].keys()
                        for m2 in overrider_marks:
                            overrider_precomp = by_starter_and_ccc[starter][lo_ccc][m2]
                            if overrider_precomp in self.long_composable_map[m1].keys():
                                # found a override replacement p''
                                replacement = self.long_composable_map[m1][overrider_precomp]
                                self.overridable_primaries[overridable][m2] = replacement
                            else:
                                s = chr(overrider_precomp) + chr(m1)
                                self.overridable_primaries[overridable][m2] = s
        for overidden in self.overridable_primaries.keys():
            for mark in self.overridable_primaries[overidden].keys():
                ccc = self.ccc_val_map[mark]
                ccc_code = self.ccc_enum_map[ccc]
                overrider = self.overridable_primaries[overidden][mark]
                override_primary = ord(overrider[0])
                if overrider in self.overridable_primaries.keys():
                    # The overrider is recursively overridable.   Make sure there
                    # is no mark in its override set that would not have already been 
                    # considered in the processing of the overridden primary itself.
                    for mark2 in self.overridable_primaries[overrider].keys():
                        ccc2 = self.ccc_val_map[mark2]
                        ccc_code2 = self.ccc_enum_map[ccc2]
                        if ccc_code2 < ccc_code and not mark2 in self.overridable_primaries[overidden].keys():
                            print("unexpected recursive override %x %x => %x %x" % (overridden, mark, overrider, mark2))

    def implement_overridable_primaries_as_long_composables(self):
        for overidden in self.overridable_primaries.keys():
            for mark in self.overridable_primaries[overidden].keys():
                overrider = self.overridable_primaries[overidden][mark]
                self.long_composable_map[mark][overidden] = overrider

    def show_overridable_primaries(self):
        for cp in sorted(self.overridable_primaries.keys()):
            decomp = ucd.decomp_map[cp]
            starter = ord(decomp[0])
            overridden_mark = ord(decomp[1])
            print("%x == %x %s" % (cp, starter, self.cp_with_ccc(overridden_mark)))
            for mark in self.overridable_primaries[cp].keys():
                overrider = self.overridable_primaries[cp][mark]
                seq = " ".join(["%x" % ord(c) for c in overrider])
                print("  %x %s ==> %s" % (cp, self.cp_with_ccc(mark), seq))

    #
    # If a given starter has precompositions with marks of
    # more than one canonical combining class (ccc), the classes
    # must be examined in separate passes to ensure that
    # precompositions are always chosen using the mark of
    # the lowest class.  
    #
    def allocate_ccc_passes(self):
        ccc_set_by_starter = {}
        ccc_conflict_map = {}
        for mark in self.long_composable_map.keys():
            ccc = self.ccc_val_map[mark]
            ccc_code = self.ccc_enum_map[ccc]
            if not ccc_code in ccc_conflict_map.keys():
                ccc_conflict_map[ccc_code] = set()
            for starter in self.long_composable_map[mark].keys():
                if not starter in ccc_set_by_starter.keys():
                    ccc_set_by_starter[starter] = {ccc_code}
                else:
                    if not ccc_code in ccc_set_by_starter[starter]:
                        #new ccc for this starter: conflict
                        for other in ccc_set_by_starter[starter]:
                            if other < ccc_code:
                                ccc_conflict_map[ccc_code].add(other)
                            else:
                                ccc_conflict_map[other].add(ccc_code)
                        ccc_set_by_starter[starter].add(ccc_code)
        for ccc in sorted(ccc_conflict_map.keys()):
            min_pass = 0
            for other in ccc_conflict_map[ccc]:
                if self.ccc_pass_allocation[other] >= min_pass:
                    min_pass = self.ccc_pass_allocation[other] + 1
                    if min_pass >= self.pass_count:
                        self.pass_count = min_pass + 1
            while len(self.pass_cccs) <= min_pass: self.pass_cccs.append([])
            self.pass_cccs[min_pass].append(ccc)
            self.ccc_pass_allocation[ccc] = min_pass

    def show_ccc_pass_allocation(self):
        print("%i passes allocated:" % self.pass_count)
        for k in sorted(self.ccc_pass_allocation.keys()):
            print("ccc = %s (%s) allocated to pass %i." % (k, self.ccc_enum_rmap[k], self.ccc_pass_allocation[k]))

    def create_conditional_mark_codes(self):
        self.conditional_mark_codes = {}
        self.max_conditional_code_bits = {}
        for pass_no in range(self.pass_count):
            self.conditional_mark_codes[pass_no] = {}
            self.max_conditional_code_bits[pass_no] = 1 # default
        pass_data = {}
        for mark in self.long_composable_map.keys():
            ccc = self.ccc_val_map[mark]
            ccc_code = self.ccc_enum_map[ccc]
            pass_code = self.ccc_pass_allocation[ccc_code]
            if not pass_code in pass_data.keys():
                pass_data[pass_code] = {}
            starter_rg_set_map = range_usets_from_cps(self.long_composable_map[mark].keys())
            for pfx_code in starter_rg_set_map.keys():
                if not pfx_code in pass_data[pass_code].keys():
                    pass_data[pass_code][pfx_code] = []
                pass_data[pass_code][pfx_code].append(mark)
        for pass_no in range(self.pass_count):
            for pfx_code in pass_data[pass_no].keys():
                mark_list = pass_data[pass_no][pfx_code]
                if len(mark_list) == 1: continue  # unconditional mark for this starter range
                if not pass_code in self.conditional_mark_codes.keys():
                    self.conditional_mark_codes[pass_no] = {}
                if not pfx_code in self.conditional_mark_codes[pass_no].keys():
                    self.conditional_mark_codes[pass_no][pfx_code] = {}
                bits = ceil_log2(len(mark_list) + 1)  # code 0 reserved
                mask = (1 << bits) - 1
                allocated = {}
                allocated[0] = 0  # code 0 is reserved 
                unassigned_marks = []
                for mark in mark_list:
                    potential_code = mark & mask
                    if potential_code in allocated:
                        unassigned_marks.append(mark)
                    else:
                        self.conditional_mark_codes[pass_no][pfx_code][mark] = potential_code
                        allocated[potential_code] = mark
                for mark in unassigned_marks:
                    for code in range(1, 1 << bits):
                        if code in allocated:
                            continue
                        self.conditional_mark_codes[pass_no][pfx_code][mark] = code
                        allocated[code] = mark
                        break
                if self.max_conditional_code_bits[pass_no] < bits:
                    self.max_conditional_code_bits[pass_no] = bits

    def show_conditional_codes(self):
        ccc_enum_rmap = ucd.property_object_map['ccc'].enum_integer_to_value_map
        for pass_no in range(self.pass_count):
            print("Pass %i, max_conditional_code_bits =  %i" % (pass_no, self.max_conditional_code_bits[pass_no]))
            for pfx_code in self.conditional_mark_codes[pass_no].keys():
                assignments = self.conditional_mark_codes[pass_no][pfx_code]
                codes = ["%x => %i" % (c, assignments[c]) for c in sorted(assignments.keys())]
                print("  pfx_code %s: %s" % (pfx_code_string(pfx_code), ", ".join(codes)))

    def generate_singleton_stage(self):
        s = singleton_header
        rg_set_map = range_usets_from_cps(self.singleton_map.keys())
        for pfx_code in rg_set_map.keys():
            code_str = pfx_code_string(pfx_code)
            scope = "b_%s" % code_str
            s += gen_nested_pfx_code(pfx_code)
            s += self.builder.open_scope(code_str, scope)
            pfx_xlate_map = {}
            singleton_list = uset_to_member_list(rg_set_map[pfx_code])
            for cp1 in singleton_list:
                pfx_xlate_map[cp1] = chr(self.singleton_map[cp1])
            bit_xfrm_sets = u8_bit_transform_sets(pfx_xlate_map)
            del_usets = u8_deletion_sets(pfx_xlate_map)
            bit_xfrm_data = install_bit_xfrm_usets(self.builder, bit_xfrm_sets)
            del_vars = install_del_usets(self.builder, del_usets)
            s += self.builder.generate_scope_compilations()
            s += generateUpdateBitXfrms(scope, bit_xfrm_data, "XfrmVar", "nullptr")
            #s += generateDel
            s += self.builder.close_scope()
        s += singleton_final_code
        return s

    def generate_excluded_composite_stage(self):
        s = excluded_composite_header
        rg_set_map = range_usets_from_cps(self.excluded_composite_map.keys())
        for pfx_code in rg_set_map.keys():
            code_str = pfx_code_string(pfx_code)
            scope = "b_%s" % code_str
            s += gen_nested_pfx_code(pfx_code)
            s += self.builder.open_scope(code_str, scope)
            pfx_xlate_map = {}
            composite_list = uset_to_member_list(rg_set_map[pfx_code])
            for cp in composite_list:
                pfx_xlate_map[cp] = self.excluded_composite_map[cp]
            bit_xfrm_sets = u8_bit_transform_sets(pfx_xlate_map)
            del_usets = u8_deletion_sets(pfx_xlate_map)
            bit_xfrm_data = install_bit_xfrm_usets(self.builder, bit_xfrm_sets)
            del_vars = install_del_usets(self.builder, del_usets)
            s += self.builder.generate_scope_compilations()
            s += generateUpdateBitXfrms(scope, bit_xfrm_data, "XfrmVar", "nullptr")
            s += self.builder.close_scope()
        s += excluded_composite_final_code
        return s

    def generate_nfc_stage(self, pass_no):
        mark_code_bits = self.max_conditional_code_bits[pass_no]
        s = gen_pass_header_code(pass_no, mark_code_bits)
        pass_data = {}
        for mark in self.long_composable_map.keys():
            ccc = self.ccc_val_map[mark]
            ccc_code = self.ccc_enum_map[ccc]
            if self.ccc_pass_allocation[ccc_code] == pass_no:
                rg_set_map = range_usets_from_cps(self.long_composable_map[mark].keys())
                for pfx_code in rg_set_map.keys():
                    if not pfx_code in pass_data.keys():
                        pass_data[pfx_code] = {}
                    pass_data[pfx_code][mark] = rg_set_map[pfx_code]
        for pfx_code in pass_data.keys():
            code_str = pfx_code_string(pfx_code)
            scope = "b_%s" % code_str
            s += gen_pass_pfx_code(pfx_code)
            starters_var = {}
            mark_var = {}
            s += self.builder.open_scope(code_str, scope)
            external_cccs = {}
            ccc_or_NR_defs = ""
            for mark in sorted(pass_data[pfx_code].keys()):
                starters_var[mark] = self.builder.install_uset(pass_data[pfx_code][mark])
                mark_var[mark] = self.builder.install_uset(singleton_uset(mark))
                ccc = self.ccc_val_map[mark] 
                if not ccc in external_cccs.keys():
                    ccc_v = "ccc_%s_%s" % (ccc, code_str)
                    self.builder.install_external_uset(ccc_v, "cccObj->GetCodepointSet(CCC_ns::%s)" % ccc)
                    external_cccs[ccc] = ccc_v
                    ccc_or_NR_defs += "    PabloAST * %s_or_NR = %s.createOr(%s, ccc_NR);\n" % (ccc_v, scope, ccc_v)
            s += self.builder.generate_scope_compilations()
            s += ccc_or_NR_defs
            for mark in sorted(pass_data[pfx_code].keys()):
                ccc_enum = self.ccc_val_map[mark]
                ccc_code = self.ccc_enum_map[ccc_enum]
                mark_code = 1 # default
                if pass_no in self.conditional_mark_codes.keys():
                    if pfx_code in self.conditional_mark_codes[pass_no].keys():
                        mark_code = self.conditional_mark_codes[pass_no][pfx_code][mark]
                s += gen_mark_case_logic(mark, starters_var[mark], mark_var[mark], ccc_enum, code_str, mark_code)
            s += self.builder.close_scope()
        s += "    PabloAST * composable2nd = markCode[0];\n"
        for i in range(1, mark_code_bits):
            s += "    composable2nd = pb.createOr(composable2nd, markCode[%i]);\n" % i
        s+=finalize_nfc_stage(pass_no)
        return s

    def generate_long_composition_xfrm_stage(self, pass_no):
        mark_code_bits = self.max_conditional_code_bits[pass_no]
        s = gen_long_composition_xfrm_header_code(pass_no, mark_code_bits)
        pass_data = {}
        for mark in self.long_composable_map.keys():
            ccc = self.ccc_val_map[mark]
            ccc_code = self.ccc_enum_map[ccc]
            if self.ccc_pass_allocation[ccc_code] == pass_no:
                rg_set_map = range_usets_from_cps(self.long_composable_map[mark].keys())
                for pfx_code in rg_set_map.keys():
                    if not pfx_code in pass_data.keys():
                        pass_data[pfx_code] = {}
                    pass_data[pfx_code][mark] = rg_set_map[pfx_code]
        bit_xfrm_data = {}
        del_vars = {}
        for pfx_code in pass_data.keys():
            code_str = pfx_code_string(pfx_code)
            scope = "b_%s" % code_str
            s += gen_long_composition_pfx_code(pfx_code)
            s += self.builder.open_scope(code_str, scope)
            bit_xfrm_data[pfx_code] = {}
            del_vars[pfx_code] = {}
            for mark in pass_data[pfx_code].keys():
                starters = uset_to_member_list(pass_data[pfx_code][mark])
                pfx_xlate_map = {}
                for cp1 in starters:
                    pfx_xlate_map[cp1] = self.long_composable_map[mark][cp1]
                bit_xfrm_sets = u8_bit_transform_sets(pfx_xlate_map)
                bit_xfrm_data[pfx_code][mark] = install_bit_xfrm_usets(self.builder, bit_xfrm_sets)
                del_usets = u8_deletion_sets(pfx_xlate_map)
                del_vars[pfx_code][mark] = install_del_usets(self.builder, del_usets)
            s += self.builder.generate_scope_compilations()
            for mark in sorted(pass_data[pfx_code].keys()):
                mark_code = 1 # default
                if pass_no in self.conditional_mark_codes.keys():
                    if pfx_code in self.conditional_mark_codes[pass_no].keys():
                        mark_code = self.conditional_mark_codes[pass_no][pfx_code][mark]
                s += "    PabloAST * foundMark_%x = %s_bnc.EQ(markCodeAtStarter, %i);\n" % (mark, scope, mark_code)
                s += "    {"
                s += generateUpdateBitXfrms(scope, bit_xfrm_data[pfx_code][mark], "XfrmVar", "foundMark_%x" % mark)
                s += "    }"
            s += self.builder.close_scope()
        s += long_composable_final_code
        return s

    def generate_long_composable_pipeline(self):
        t = string.Template(long_composable_pipeline_step_template)
        t2 = string.Template(long_composable_pipeline_template)
        logic = ""
        input_basis = "Basis"
        for pass_no in range(self.pass_count):

            if pass_no == self.pass_count - 1:
                output_basis = "FinalBasis"
            else:
                output_basis = "XfrmedBasis%i" % pass_no
                logic += "    StreamSet * %s = P.CreateStreamSet(8, 1);\n" % output_basis
            mark_code_bits = self.max_conditional_code_bits[pass_no]
            logic += t.substitute(pass_no = pass_no, 
                                  step = pass_no,
                                  mark_code_bits = mark_code_bits, 
                                  input_basis = input_basis,
                                  output_basis = output_basis)
            input_basis = output_basis # for next pass
            if pass_no == 3:
                output_basis = "XfrmedBasis%iA" % pass_no
                logic += "    StreamSet * %s = P.CreateStreamSet(8, 1);\n" % output_basis
                logic += t.substitute(pass_no = pass_no,
                                      step = "%iA" % pass_no,
                                      mark_code_bits = mark_code_bits,
                                      input_basis = input_basis,
                                      output_basis = output_basis)
            input_basis = output_basis # for next pass
        return t2.substitute(pipeline_logic = logic, final_pass_no = self.pass_count - 1)

    def generate_self_composable_CC_stage(self):
        s = self_composable_CC_header
        self.builder.open_scope("", "pb")
        self_composable_usets = []
        for A in self.self_composables.keys():
            AA = self.self_composables[A]
            self_composable_usets.append(self.builder.install_uset(singleton_uset(A)))
            self_composable_usets.append(self.builder.install_uset(singleton_uset(AA)))
        s += self.builder.generate_scope_compilations()
        s += "    std::vector<PabloAST *> self_composables(%i);\n" % len(self_composable_usets)
        for i in range(len(self_composable_usets)):
            s += "    self_composables[%i] = %s;\n" % (i, self_composable_usets[i])
        s += self_composable_CC_final_code
        return s

    def generate_self_composable_stage(self):
        s = self_composable_header
        t = string.Template(self_composable_case_template)
        self.builder.open_scope("", "pb")
        i = 0
        s += self.builder.generate_scope_compilations()
        for A in self.self_composables.keys():
            AA = self.self_composables[A]
            xfrm_map = {A : chr(AA), AA : chr(A)}
            A_len = u8_encoded_length(A)
            AA_len = u8_encoded_length(AA)
            A_AST = "self_composable_CCs[%i]" % i
            AA_AST = "self_composable_CCs[%i]" % (i + 1)
            s += t.substitute(A = "%x" % A, A_len = A_len, AA_len = AA_len, A_AST = A_AST, AA_AST = AA_AST)
            scope = "b_%x" % A
            s += self.builder.open_scope("x%x" % A, scope)
            bit_xfrm_sets = u8_bit_transform_sets(xfrm_map)
            bit_xfrm_data = install_bit_xfrm_usets(self.builder, bit_xfrm_sets)
            del_usets = u8_deletion_sets(xfrm_map)
            del_vars = install_del_usets(self.builder, del_usets)
            s += self.builder.generate_scope_compilations()
            s += generateUpdateBitXfrms("b_%x" % A, bit_xfrm_data, "XfrmVar", "xfrm_%x" % A)
            s += self.builder.close_scope()
            i += 2
        s += self_composable_final_code
        return s

    def generate_short_composable_stage(self):
        s = short_composable_header
        rg_set_map = range_usets_from_cps(self.short_composable_map.keys())
        for pfx_code in rg_set_map.keys():
            code_str = pfx_code_string(pfx_code)
            scope = "b_%s" % code_str
            s += gen_short_composable_pfx_code(pfx_code)
            s += self.builder.open_scope(code_str, scope)
            generated = {}
            generated_seconds = {}
            cp1_list = uset_to_member_list(rg_set_map[pfx_code])
            for cp1 in cp1_list:
                for cp2 in self.short_composable_map[cp1].keys():
                    if not cp2 in generated_seconds.keys():
                        generated_seconds[cp2] = self.builder.install_uset(singleton_uset(cp2), pfx_code_lgth(pfx_code))
            s += self.builder.generate_scope_compilations("LookAhead", pfx_code_lgth(pfx_code))
            # Have finished compiling all seconds with lookaheads
            generated = {}
            cp1_list = uset_to_member_list(rg_set_map[pfx_code])
            for cp1 in cp1_list:
                if not cp1 in generated.keys():
                    generated[cp1] = self.builder.install_uset(singleton_uset(cp1))
            s += self.builder.generate_scope_compilations()
            case_code = {}
            deferred = []
            for cp1 in cp1_list:
                for cp2 in self.short_composable_map[cp1].keys():
                    precomposed = self.short_composable_map[cp1][cp2]
                    case_code[(cp1, cp2)] = generate_short_case_code(self.builder, code_str, cp1, cp2, generated, generated_seconds, precomposed)
                    if precomposed in cp1_list:
                        deferred.append(precomposed)
                        print("Deferring %x" % precomposed)
            precomposed_generated = {}
            for cp1 in cp1_list:
                if not cp1 in deferred:
                    for cp2 in self.short_composable_map[cp1].keys():
                        s += case_code[(cp1, cp2)]
                        precomp = self.short_composable_map[cp1][cp2]
                        found = "found_%x_%x" % (cp1, cp2)
                        precomposed_generated[precomp] = found
            while len(deferred) > 0:
                next_deferred = []
                for cp1 in deferred:
                    if cp1 in precomposed_generated.keys():
                        s += "    %s.createAssign(%s, %s.createOr(%s, %s));\n" % (scope, generated[cp1], scope, generated[cp1], precomposed_generated[cp1])
                        for cp2 in self.short_composable_map[cp1].keys():
                            s += case_code[(cp1, cp2)]
                            precomp = self.short_composable_map[cp1][cp2]
                            precomposed_generated[precomp] = found
                    else: next_deferred.append(cp1)
                deferred = next_deferred
            s += finalize_short_composable_pfx_code(pfx_code)
            s += self.builder.close_scope()
        s += short_composable_final_code
        return s

    def generate_u8_insertion_kernel(self):
        insertion_usets = insert_map_to_bixnum_usets(self.max_insert_map)
        bixnum_bits = len(insertion_usets)
        t = string.Template(u8_insertion_bixnum_template)
        s = t.substitute(insertion_bixnum_bits = bixnum_bits)
        self.builder.open_scope("", "pb")
        insertions = []
        for i in range(bixnum_bits):
            insertions.append(self.builder.install_uset(insertion_usets[i]))
        s += self.builder.generate_scope_compilations("Advance")
        for i in range(bixnum_bits):
            s += "    insertions[%s] = %s;\n" % (i, insertions[i])
        s += u8_insertion_final_code
        return s

    #  Generate the two character classes which identify characters
    #  that may be involved in NFC conversion.
    #  (1)  candidates which are marks, generate marks (nonstarter decomposition),
    #       or otherwise potentially composable with a prior character
    #       (Hangul V and T type characters, second characters of short composables)
    #  (2)  characters with required expansions (singletons, excluded composites)
    def generate_candidate_classes(self):
        t = string.Template(candidate_class_template)
        # non-starter decompositions are either marks or generate marks
        candidate_class = self.non_starter_uset
        candidate_class = uset_union(candidate_class, self.composable_seconds)
        for ccc_enum in self.ucd.ccc_map.keys():
            if ccc_enum != 'NR':  # 
                candidate_class = uset_union(candidate_class, self.ucd.ccc_map[ccc_enum])
        for cp in self.self_composables.keys():
            doubleton = self.self_composables[cp]
            candidate_class = uset_union(candidate_class, singleton_uset(doubleton))
        # Include the Hangul V and T composables
        VBase = 0x1161
        VCount = 21
        TBase = 0x11A7
        TCount = 28
        candidate_class = uset_union(candidate_class, range_uset(VBase, VBase + VCount - 1))
        candidate_class = uset_union(candidate_class, range_uset(TBase, TBase + TCount - 1))
        candidate_uset = self.builder.install_uset(candidate_class)

        expansion_required_class = empty_uset()
        #for cp in self.overridable_primaries.keys():
        #    expansion_required_class = uset_union(expansion_required_class, singleton_uset(cp))
        for cp in self.singleton_map.keys():
            expansion_required_class = uset_union(expansion_required_class, singleton_uset(cp))
        for cp in self.excluded_composite_map.keys():
            expansion_required_class = uset_union(expansion_required_class, singleton_uset(cp))
        expansion_required_uset = self.builder.install_uset(expansion_required_class)
        return t.substitute(candidate_uset = candidate_uset, expansion_required_uset = expansion_required_uset)

    def emit_normalization_cpp(self):
        basename = 'normalization-generated'
        f = cformat.open_cpp_file_for_write(basename)
        kernels = ""
        kernels += self.generate_u8_insertion_kernel()
        kernels += self.generate_singleton_stage()
        kernels += self.generate_excluded_composite_stage()
        kernels += self.generate_self_composable_CC_stage()
        kernels += self.generate_self_composable_stage()
        kernels += self.generate_short_composable_stage()
        for pass_no in range(5):
            kernels += self.generate_nfc_stage(pass_no)
            kernels += self.generate_long_composition_xfrm_stage(pass_no)
        kernels += self.generate_long_composable_pipeline()
        kernels += self.generate_candidate_classes()
        uset_definitions = self.builder.get_uset_definitions()
        t = string.Template(nfc_generated_cpp_template)
        s = t.substitute(uset_declarations = uset_definitions, kernels = kernels)
        f.write(s)
        cformat.close_cpp_file(f)

if __name__ == "__main__":
    ucd = UCD_database()
    generator = NFC_generator(ucd)
    generator.create_mappings()
    generator.add_doubleton_shorts()
    generator.add_overridable_seconds()
    generator.determine_overridable_primaries()
    generator.create_max_insert_map()
    generator.allocate_ccc_passes()
    generator.implement_overridable_primaries_as_long_composables()
    generator.create_conditional_mark_codes()
    #generator.show_overridable_primaries()
    #generator.show_ccc_pass_allocation()
    #generator.show_conditional_codes()
    generator.display_singletons()
    #generator.display_short_composables()
    #generator.display_self_composables()
    #generator.display_long_composables()
    #generator.display_excluded_composites()
    #generator.display_recursive_decomposables()
    #generator.display_max_insert_map()
    generator.emit_normalization_cpp()
