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

scoped_compiler_template = r"""    UTF::UTF_Compiler ${pfx}_compiler(getInputStreamVar("Basis"), ${scope}, pablo::BitMovementMode::${movement});
    std::vector<Var *> ${pfx}_vars(${num_roles});
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

    def install_uset(self, uset):
        key = uset_structural_key(uset)
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

    def generate_scope_compilations(self, movement_mode = "LookAhead"):
        pfx = ""
        if movement_mode != "LookAhead":
            pfx = pfx + "_adv"
        t1 = string.Template(scoped_compiler_template)
        t2 = string.Template(assignment_template)
        t3 = string.Template(external_template)
        assigs = ""
        for i in range(len(self.current_scope_roles)):
            k = self.current_scope_roles[i]
            if k in self.installed_usets.keys():
                assigs += t2.substitute(pfx = pfx, scope = self.current_scope, index = i, role = k)
            else:
                expr = self.external_usets[k]
                assigs += t3.substitute(pfx = pfx, scope = self.current_scope, index = i, role = k, external_expr = expr)
        defs = t1.substitute(pfx = pfx,
                             scope = self.current_scope,
                             movement = movement_mode,
                             num_roles = len(self.current_scope_roles),
                             assignments = assigs)
        return defs

    def close_scope(self):
        return "    }\n"


def u8_adjustment_bixnums_from_codepoint_map(translation_map):
    for cp1 in translation_map.keys():
        cp2 = translation_map[cp1]
        ldiff = u8_encoded_length(cp2) - u8_encoded_length(cp1)
        insertion_bixnum_usets = []
        deletion_bixnum_usets = []
        if ldiff > 0:
            bit = 0
            while ldiff > 0:
                if len(insertion_bixnum_usets) <= bit:
                    insertion_bixnum_usets.append(empty_uset())
                if ldiff & 1 == 1:
                    insertion_bixnum_usets[bit] = uset_union(insertion_bixnum_usets[bit], singleton_uset(cp1))
                ldiff >>=  1
                bit += 1
        elif ldiff < 0:
            ldiff = -ldiff
            bit = 0
            while ldiff > 0:
                if len(deletion_bixnum_usets) <= bit:
                    deletion_bixnum_usets.append(empty_uset())
                if ldiff & 1 == 1:
                    deletion_bixnum_usets[bit] = uset_union(deletion_bixnum_usets[bit], singleton_uset(cp1))
                ldiff >>=  1
                bit += 1
    return (insertion_bixnum_usets, deletion_bixnum_usets)

def u8_deletion_usets_from_codepoint_map(translation_map):
    deletion_usets = {}
    for cp1 in translation_map.keys():
        cp2 = translation_map[cp1]
        len1 = u8_encoded_length(cp1)
        len2 = u8_encoded_length(cp2)
        if len1 > len2:
            ldiff = len1 - len2
            if not ldiff in deletion_usets.keys():
                deletion_usets[ldiff] = empty_uset()
            deletion_usets[ldiff] = uset_union(deletion_usets[ldiff], singleton_uset(cp1))
    return deletion_usets


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
        return "0_7F"
    if pfx_code <= 0xDF:
        return "%x_%x" % (pfx_code, pfx_code | 0x03)
    return "%x" % pfx_code

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
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * SelectMask, StreamSet * XfrmBasis)
: PabloKernel(ts, "SingletonCanonicalization" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"SelectMask", SelectMask}, Binding{"XfrmBasis", XfrmBasis}}) {}

void SingletonCanonicalization::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    // DeleteVar will be inverted to produce SelectMask
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

singleton_case_template = r"""//     ${cp} => ${canon}
${case_logic}
"""

mark_deletion_template = r"""    del_${code_str} = b_${code_str}.createOr(del_${code_str}, ${del_marker});
"""

def generate_singleton_code(code_str, cp, canon, cp_var, Num0s, ToDel):
    xlate_code = CharacterTranslationLogic(cp, canon, cp_var, "xfrm_%s" % code_str, "b_%s" % code_str)
    if ToDel > 0:
        t = string.Template(mark_deletion_template)
        xlate_code += t.substitute(code_str = code_str, del_marker = cp_var)
        for i in range(1, ToDel):
            adv_mrkr = "b_%s.createAdvance(%s, %i)" % (code_str, cp_var, i)
            xlate_code += t.substitute(code_str = code_str, del_marker = adv_mrkr)
    t = string.Template(singleton_case_template)
    s = t.substitute(code_str = code_str,
                        cp = "%x" % cp,
                        canon = "%x" % canon,
                        case_logic = xlate_code)
    return s

nested_pfx_final_template = r"""
    for (unsigned i = 0; i < 8; i++) {
        ${builder}.createAssign(XfrmVar[i], $builder.createOr(XfrmVar[i], xfrm_${code_str}[i]));
    }
"""

update_del_var_template = r"""
    ${builder}.createAssign(DeleteVar, $builder.createOr(DeleteVar, del_${code_str}));
"""

def finalize_nested_pfx_code(pfx_code, has_del):
    code_str = pfx_code_string(pfx_code)
    scope = "b_%s" % code_str
    t1 = string.Template(nested_pfx_final_template)
    s = t1.substitute(builder = scope, code_str = code_str)
    if has_del:
        t2 = string.Template(update_del_var_template)
        s2 = t2.substitute(builder = scope, code_str = code_str)
        s += s2
    return s


singleton_final_code = r"""
    Var * XfrmOutputVar = getOutputStreamVar("XfrmBasis");
    for (unsigned i = 0; i < 8; i++) {
        Var * xfrm_out = pb.createExtract(XfrmOutputVar, pb.getInteger(i));
        //  pb.createAssign(xfrm_out, XfrmVar[i]);
        pb.createAssign(xfrm_out, pb.createXor(Basis[i], XfrmVar[i]));
    }
    Var * MaskOutputVar = pb.createExtract(getOutputStreamVar("SelectMask"), pb.getInteger(0));
    pb.createAssign(MaskOutputVar, pb.createInFile(pb.createNot(DeleteVar)));
}
"""

excluded_composite_header = r"""//
ExcludedCompositeStage::ExcludedCompositeStage
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * SelectMask, StreamSet * XfrmBasis)
: PabloKernel(ts, "ExcludedCompositeStage" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"SelectMask", SelectMask}, Binding{"XfrmBasis", XfrmBasis}}) {}

void ExcludedCompositeStage::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    // DeleteVar will be inverted to produce SelectMask
    Var * DeleteVar = pb.createVar("DeleteVar", All0);
    std::vector<Var *> XfrmVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        XfrmVar[i] = pb.createVar("XfrmBasis" + std::to_string(i), All0);
    }
"""

excluded_composite_final_code = r"""
    Var * XfrmOutputVar = getOutputStreamVar("XfrmBasis");
    for (unsigned i = 0; i < 8; i++) {
        Var * xfrm_out = pb.createExtract(XfrmOutputVar, pb.getInteger(i));
        //  pb.createAssign(xfrm_out, XfrmVar[i]);
        pb.createAssign(xfrm_out, pb.createXor(Basis[i], XfrmVar[i]));
    }
    Var * MaskOutputVar = pb.createExtract(getOutputStreamVar("SelectMask"), pb.getInteger(0));
    pb.createAssign(MaskOutputVar, pb.createInFile(pb.createNot(DeleteVar)));
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
{Binding{"Basis", Basis}, Binding{"ccc_NR", ccc_NR}},
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
    pb.createIf(pb.createAnd(${pfx_test_var}, mark_ahead_${mark_lookahead}), ${builder});
"""

def gen_pass_pfx_code(pfx_code):
    code_str = pfx_code_string(pfx_code)
    pfx_test_var = "pfx_%s_test" % code_str
    test = prefix_test_logic(pfx_code)
    scope = "b_%s" % code_str
    t = string.Template(pass_pfx_template)
    return t.substitute(builder = scope,
                        pfx_test_var = pfx_test_var,
                        mark_lookahead = pfx_code_lgth(pfx_code),
                        logic = test)

mark_case_template = r"""
//  Case for mark ${mark}
    PabloAST * ${builder}_possible_${mark}_pos = ${builder}.createScanTo(${mark_starters}, ${ccc_v}_or_NR);
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
    Var * indexVar = getOutputStreamVar("Index_ccc_NR_or_MarkFound");
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
        (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * MarkCode,
                                       StreamSet * OutputBasis);
protected:
    void generatePabloMethod() override;
};

ApplyLongComposition${pass_no}::ApplyLongComposition${pass_no}
    (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * MarkCode,
                                   StreamSet * OutputBasis)
: PabloKernel(ts, "ApplyLongComposition${pass_no}_" + Basis->shapeString(),
{Binding{"Basis", Basis}, Binding{"MarkCode", MarkCode}},
{Binding{"OutputBasis", OutputBasis}}) {}

void ApplyLongComposition${pass_no}::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    std::vector<PabloAST *> markCode = getInputStreamSet("MarkCode");
    const unsigned markCodeBits = ${markCodeBits};
    PabloAST * markFound = markCode[0];
    for (unsigned i = 1; i < markCodeBits; i++) {
        markFound = pb.createOr(markFound, markCode[i]);
    }
    std::vector<Var *> XfrmVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        XfrmVar[i] = pb.createVar("XfrmBasis" + std::to_string(i), All0);
    }
    Var * DeleteVar = pb.createVar("DeleteVar", All0);
"""

def gen_long_composition_xfrm_header_code(pass_no, mark_code_bits):
    s = string.Template(long_composable_application_template)
    return s.substitute(pass_no = pass_no,
                        markCodeBits = mark_code_bits)

long_composition_pfx_template = r"""
    auto ${builder} = pb.createScope();
    PabloAST * ${pfx_test_var} = ${logic};
    pb.createIf(pb.createAnd(${pfx_test_var}, markFound), ${builder});
    std::vector<PabloAST *> xfrm_${code_str}(8, All0);
    PabloAST * del_${code_str} = All0;
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

long_composable_pfx_final_template = r"""
    for (unsigned i = 0; i < 8; i++) {
        ${builder}.createAssign(XfrmVar[i], $builder.createOr(XfrmVar[i], xfrm_${code_str}[i]));
    }
    ${builder}.createAssign(DeleteVar, $builder.createOr(DeleteVar, del_${code_str}));
"""

def finalize_long_composable_pfx_code(pfx_code):
    code_str = pfx_code_string(pfx_code)
    scope = "b_%s" % code_str
    t = string.Template(long_composable_pfx_final_template)
    return t.substitute(builder = scope,
                        code_str = code_str)



short_composable_header = r"""//
ShortComposableTranslation::ShortComposableTranslation
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * DeletePrior, StreamSet * XfrmBasis)
: PabloKernel(ts, "ShortComposableTranslation" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"DeletePrior", DeletePrior}, Binding{"XfrmBasis", XfrmBasis}}) {}

void ShortComposableTranslation::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    Var * DeletePriorVar = pb.createVar("DeletePriorVar", All0);
    std::vector<Var *> XfrmVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        XfrmVar[i] = pb.createVar("XfrmBasis" + std::to_string(i), All0);
    }
"""

short_composable_starter_template = r"""//  Cases for ${cp1}
    PabloAST * after_${cp1} = ${builder}.createAdvance(${cp1_var}, ${cp1_len});
"""

def generate_short_case_starter_code(builder, cp1, generated):
    t = string.Template(short_composable_starter_template)
    s = t.substitute(builder = builder,
                        cp1 = "%x" % cp1,
                        cp1_var = generated[cp1],
                        cp1_len=u8_encoded_length(cp1))
    return s

def generate_short_case_starter_code2(builder, cp1, generated):
    t = string.Template(short_composable_starter_template)
    s = t.substitute(builder = builder,
                        cp1 = "%x" % cp1,
                        cp1_var = generated[cp1],
                        cp1_len=u8_encoded_length(cp1))
    return s

short_composable_case_template = r"""//     ${cp1} + ${cp2} => ${precomposed}
    PabloAST * found_${cp1}_${cp2} = b_${code_str}.createAnd(after_${cp1}, ${cp2_var});
    del_prior_${code_str} = b_${code_str}.createOr(del_prior_${code_str}, found_${cp1}_${cp2});
"""

def generate_short_case_code(code_str, cp1, cp2, generated, precomposed):
    t = string.Template(short_composable_case_template)
    s = t.substitute(code_str = code_str,
                        cp1 = "%x" % cp1,
                        cp2 = "%x" % cp2,
                        cp1_var = generated[cp1],
                        cp2_var = generated[cp2],
                        cp1_len=u8_encoded_length(cp1),
                        precomposed = "%x" % precomposed)
    xlate_code = CharacterTranslationLogic(cp2, precomposed, "found_%x_%x" % (cp1, cp2), "xfrm_%s" % code_str, "b_%s" % code_str)
    #xlate_code = add_u8_bit_translation_case(code_str, cp2, precomposed, "found_%x_%x" % (cp1, cp2))
    return s + xlate_code

short_composable_pfx_template = r"""
    auto ${builder} = pb.createScope();
    PabloAST * ${pfx_test_var} = ${logic};
    pb.createIf(${pfx_test_var}, ${builder});
    std::vector<PabloAST *> xfrm_${code_str}(8, All0);
    PabloAST * del_prior_${code_str} = All0;
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
    ${builder}.createAssign(DeletePriorVar, $builder.createOr(DeletePriorVar, del_prior_${code_str}));
"""

def finalize_short_composable_pfx_code(pfx_code):
    code_str = pfx_code_string(pfx_code)
    scope = "b_%s" % code_str
    t = string.Template(short_composable_pfx_final_template)
    return t.substitute(builder = scope,
                        code_str = code_str)

short_composable_final_code = r"""
    Var * DeleteOutputVar = getOutputStreamVar("DeletePrior");
    pb.createAssign(pb.createExtract(DeleteOutputVar, pb.getInteger(0)), DeletePriorVar);
    Var * XfrmOutputVar = getOutputStreamVar("XfrmBasis");
    for (unsigned i = 0; i < 8; i++) {
        Var * xfrm_out = pb.createExtract(XfrmOutputVar, pb.getInteger(i));
        pb.createAssign(xfrm_out, XfrmVar[i]);
    }
}
"""

#
#  UTF-8 character translation using Xor method for cp1 -> cp2.
#  Assumptions:
#    - marker is a bit stream marking the first byte of any cp1
#    - if u8_encoded_length(cp2) > u8_encoded_length(cp1):
#      zeroes have been inserted after the last byte of cp1
#    - if u8_encoded_length(cp2) < u8_encoded_length(cp1):
#      excess = u8_encoded_length(cp1) - u8_encoded_length(cp2)
#      The first excess positions are don't cares and will
#      ultimately be deleted
#    - pb is the PabloBuilder for logic
#
def CharacterTranslationLogic(cp1, cp2, marker, basis_var, pb):
    s = ""
    len1 = u8_encoded_length(cp1)
    len2 = u8_encoded_length(cp2)
    excess = len1 - len2
    for i in range(len2):
        # position for cp2 byte
        cp2_byte = u8_code_unit(cp2, i + 1)
        pos = excess + i
        cp1_byte = 0 # default for positions corresponding to inserted zeroes
        if pos < len1:
            cp1_byte = u8_code_unit(cp1, pos + 1)
        if cp1_byte != cp2_byte:
            diff = cp1_byte ^ cp2_byte
            # advance marker to pos
            if pos == 0:
                s += "    PabloAST * m_%x_%x_0 = %s;\n" % (cp1, cp2, marker)
            else:
                s += "    PabloAST * m_%x_%x_%i = %s.createAdvance(%s, %i);\n" % (cp1, cp2, i, pb, marker, pos)
            # apply xor logic for each bit difference between cp1_byte, cp2_byte
            for j in range(8):
                bv = "%s[%i]" % (basis_var, j)
                if ((diff >> j) & 1) == 1:
                    s += "    %s = %s.createOr(%s, m_%x_%x_%i);\n" % (bv, pb, bv, cp1, cp2, i)
    return s

#
#  UTF-8 character insertion for a given cp.
#  Assumptions:
#    - marker is a bit stream marking the first byte of any cp
#    - zero bytes have been inserted the each byte position of cp
#    - pb is the PabloBuilder for logic
#
def CharacterInsertionLogic(cp, marker, basis_var, pb):
    s = ""
    cp_len = u8_encoded_length(cp)
    for i in range(cp_len):
        cp_byte = u8_code_unit(cp, i + 1)
        if i == 0:
            s += "    PabloAST * m_%x_0 = %s;\n" % (cp, marker)
        else:
            s += "    PabloAST * m_%x_%i = %s.createAdvance(%s, %i);\n" % (cp, i, pb, marker, i)
        # apply xor logic for each bit difference between cp1_byte, cp2_byte
        for j in range(8):
            bv = "%s[%i]" % (basis_var, j)
            if ((cp_byte >> j) & 1) == 1:
                s += "    %s = %s.createOr(%s, m_%x_%i);\n" % (bv, pb, bv, cp, i)
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
            if not ldiff in deletion_usets.keys():
                deletion_usets[ldiff] = empty_uset()
            deletion_usets[ldiff] = uset_union(deletion_usets[ldiff], cp_uset)
    return deletion_usets

def u8_bit_transform_sets(char2string_map):
    bit_xfrm_sets = {}
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
        excess = 0
        if len1 > len2:
            excess = len1 - len2
        for i in range(len2):
            xlat_byte = str_bytes[i]
            pos = excess + i # position relative to cp1 starter
            cp_byte = 0 # default for positions corresponding to inserted zeroes
            if pos < len1:
                cp_byte = u8_code_unit(cp, pos + 1)
            if cp_byte != xlat_byte:
                diff = cp_byte ^ xlat_byte
                if not pos in bit_xfrm_sets.keys():
                    bit_xfrm_sets[pos] = {}
                for j in range(8):
                    if (diff >> j) & 1 == 1:
                        if not j in bit_xfrm_sets[pos].keys():
                            bit_xfrm_sets[pos][j] = empty_uset()
                        bit_xfrm_sets[pos][j] = uset_union(bit_xfrm_sets[pos][j], cp_uset)
    return bit_xfrm_sets

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


nfc_generated_cpp_template = r"""#include <kernel/unicode/normalization.h>
#include <unicode/core/unicode_set.h>
#include <unicode/data/PropertyObjectTable.h>
#include <unicode/algo/normalization.h>
#include <unicode/utf/utf_compiler.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>

using namespace pablo;
using namespace kernel;
using namespace llvm;
using namespace UCD;

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

    def create_mappings(self):
        self.singleton_map = {}
        self.short_composable_map = {}
        self.long_composable_map = {}
        self.excluded_composite_map = {}
        for precomposed in ucd.decomp_map.keys():
            if uset_member(ucd.dt_map['Can'], precomposed):
                decomp = ucd.decomp_map[precomposed]
                if len(decomp) == 1:
                    self.singleton_map[precomposed] = ord(decomp[0])
                elif len(decomp) == 2:
                    cp1 = ord(decomp[0])
                    cp2 = ord(decomp[1])
                    if self.excluded_composite(precomposed):
                        while self.excluded_composite(cp1):
                            decomp = ucd.decomp_map[cp1] + decomp[1:]
                            cp1 = ord(decomp[0])
                        self.excluded_composite_map[precomposed] = decomp
                    else:
                        if uset_member(ucd.ccc_map['NR'], cp2):
                            # Decomposition to two consecutive starters
                            if not cp1 in self.short_composable_map.keys():
                                # index by the first character of decomposition
                                self.short_composable_map[cp1] = {}
                            self.short_composable_map[cp1][cp2] = precomposed
                        else:
                            if not cp2 in self.long_composable_map.keys():
                                # index by the mark (second char of decomposition)
                                self.long_composable_map[cp2] = {}
                            self.long_composable_map[cp2][cp1] = precomposed
                else:
                    raise Exception("Unexpected: decomposition length(%x) = %i" % (precomposed, len(decomp)))

    def cp_with_ccc(self, cp):
        return "%x(%s)" % (cp, self.ccc_val_map[cp])

    def display_singletons(self):
        for cp in sorted(self.singleton_map.keys()):
            canon_cp = self.singleton_map[cp]
            print("%s => %s" % (self.cp_with_ccc(cp), self.cp_with_ccc(canon_cp)))

    def display_excluded_composites(self):
        for cp in sorted(self.excluded_composite_map.keys()):
            decomp = self.excluded_composite_map[cp]
            print("%s => %s" % (self.cp_with_ccc(cp), " ".join([self.cp_with_ccc(ord(c)) for c in decomp])))

    def display_short_composables(self):
        for cp1 in sorted(self.short_composable_map.keys()):
            print("%s: " % self.cp_with_ccc(cp1))
            for cp2 in sorted(self.short_composable_map[cp1].keys()):
                cp = self.short_composable_map[cp1][cp2]
                print("    + %s => %s" % (self.cp_with_ccc(cp2), self.cp_with_ccc(cp)))

    def display_long_composables(self):
        by_starter = {}
        for cp2 in sorted(self.long_composable_map.keys()):
            for cp1 in self.long_composable_map[cp2].keys():
                cp = self.long_composable_map[cp2][cp1]
                if not cp1 in by_starter.keys():
                    by_starter[cp1] = {}
                by_starter[cp1][cp2] = cp
        for cp1 in sorted(by_starter.keys()):
            print("%s: " % self.cp_with_ccc(cp1))
            for cp2 in sorted(by_starter[cp1].keys()):
                cp = by_starter[cp1][cp2]
                print("    + %s => %s" % (self.cp_with_ccc(cp2), self.cp_with_ccc(cp)))

    def display_recursive_decomposables(self):
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
                ldiff = u8_encoded_length(precomposed) - u8_encoded_length(starter)
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

    def display_max_insert_map(self):
        for cp in sorted(self.max_insert_map.keys()):
            print("%04x - insert %i" % (cp, self.max_insert_map[cp]))

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
            del_vars = {}
            for del_amt in del_usets.keys():
                del_vars[del_amt] = self.builder.install_uset(del_usets[del_amt])
            by_pos = {}
            for pos in sorted(bit_xfrm_sets.keys()):
                by_pos[pos] = {}
                for bit in bit_xfrm_sets[pos].keys():
                    by_pos[pos][bit] = self.builder.install_uset(bit_xfrm_sets[pos][bit])
            s += self.builder.generate_scope_compilations()
            for pos in sorted(bit_xfrm_sets.keys()):
                adv_markers = {}
                for bit in bit_xfrm_sets[pos].keys():
                    marker = by_pos[pos][bit]
                    if pos > 0:
                        if not marker in adv_markers.keys():
                            adv = "%s_adv%i" % (marker, pos)
                            s += "    PabloAST * %s = %s.createAdvance(%s, %i);\n" % (adv, scope, marker, pos)
                            adv_markers[marker] = adv
                        marker = adv_markers[marker]
                    s += "    xfrm_%s[%i] = %s.createOr(xfrm_%s[%i], %s);\n" % (code_str, bit, scope, code_str, bit, marker)
            for del_amt in sorted(del_usets.keys()):
                s += "    del_%s = %s.createOr(del_%s, %s);\n" % (code_str, scope, code_str, del_vars[del_amt])
                for d in range(1, del_amt):
                    s += "    del_%s = %s.createOr(del_%s, %s.createAdvance(%s, %i));\n" % (code_str, scope, code_str, scope, del_vars[del_amt], d)
            s += finalize_nested_pfx_code(pfx_code, len(del_usets.keys()) > 0)
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
            del_vars = {}
            for del_amt in del_usets.keys():
                del_vars[del_amt] = self.builder.install_uset(del_usets[del_amt])
            by_pos = {}
            for pos in sorted(bit_xfrm_sets.keys()):
                by_pos[pos] = {}
                for bit in bit_xfrm_sets[pos].keys():
                    by_pos[pos][bit] = self.builder.install_uset(bit_xfrm_sets[pos][bit])
            s += self.builder.generate_scope_compilations()
            for pos in sorted(bit_xfrm_sets.keys()):
                adv_markers = {}
                for bit in bit_xfrm_sets[pos].keys():
                    marker = by_pos[pos][bit]
                    if pos > 0:
                        if not marker in adv_markers.keys():
                            adv = "%s_adv%i" % (marker, pos)
                            s += "    PabloAST * %s = %s.createAdvance(%s, %i);\n" % (adv, scope, marker, pos)
                            adv_markers[marker] = adv
                        marker = adv_markers[marker]
                    s += "    xfrm_%s[%i] = %s.createOr(xfrm_%s[%i], %s);\n" % (code_str, bit, scope, code_str, bit, marker)
            for del_amt in sorted(del_usets.keys()):
                s += "    del_%s = %s.createOr(del_%s, %s);\n" % (code_str, scope, code_str, del_vars[del_amt])
                for d in range(1, del_amt):
                    s += "    del_%s = %s.createOr(del_%s, %s.createAdvance(%s, %i));\n" % (code_str, scope, code_str, scope, del_vars[del_amt], d)
            s += finalize_nested_pfx_code(pfx_code, len(del_usets.keys()) > 0)
            s += self.builder.close_scope()
        s += excluded_composite_final_code
        return s

    def generate_nfc_stage(self, pass_no):
        mark_code_bits = self.max_conditional_code_bits[pass_no]
        s = gen_pass_header_code(pass_no, mark_code_bits)
        pass_data = {}
        pfx_lgths = []
        for mark in self.long_composable_map.keys():
            ccc = self.ccc_val_map[mark]
            ccc_code = self.ccc_enum_map[ccc]
            if self.ccc_pass_allocation[ccc_code] == pass_no:
                rg_set_map = range_usets_from_cps(self.long_composable_map[mark].keys())
                for pfx_code in rg_set_map.keys():
                    if not pfx_code in pass_data.keys():
                        pass_data[pfx_code] = {}
                        pfx_lgth = pfx_code_lgth(pfx_code)
                        if not pfx_lgth in pfx_lgths:
                            pfx_lgths.append(pfx_lgth)
                    pass_data[pfx_code][mark] = rg_set_map[pfx_code]
        for pfx_lgth in sorted(pfx_lgths):
            s += "    PabloAST * mark_ahead_%i = pb.createNot(pb.createLookahead(ccc_NR, %i));\n" % (pfx_lgth, pfx_lgth)
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
        pfx_lgths = []
        for mark in self.long_composable_map.keys():
            ccc = self.ccc_val_map[mark]
            ccc_code = self.ccc_enum_map[ccc]
            if self.ccc_pass_allocation[ccc_code] == pass_no:
                rg_set_map = range_usets_from_cps(self.long_composable_map[mark].keys())
                for pfx_code in rg_set_map.keys():
                    if not pfx_code in pass_data.keys():
                        pass_data[pfx_code] = {}
                        pfx_lgth = pfx_code_lgth(pfx_code)
                        if not pfx_lgth in pfx_lgths:
                            pfx_lgths.append(pfx_lgth)
                    pass_data[pfx_code][mark] = rg_set_map[pfx_code]
        del_vars = {}
        by_pos = {}
        for pfx_code in pass_data.keys():
            code_str = pfx_code_string(pfx_code)
            scope = "b_%s" % code_str
            s += gen_long_composition_pfx_code(pfx_code)
            s += self.builder.open_scope(code_str, scope)

            del_vars[pfx_code] = {}
            by_pos[pfx_code] = {}
            for mark in pass_data[pfx_code].keys():
                starters = uset_to_member_list(pass_data[pfx_code][mark])
                pfx_xlate_map = {}
                for cp1 in starters:
                    pfx_xlate_map[cp1] = chr(self.long_composable_map[mark][cp1])
                bit_xfrm_sets = u8_bit_transform_sets(pfx_xlate_map)
                del_usets = u8_deletion_sets(pfx_xlate_map)
                del_vars[pfx_code][mark] = {}
                for del_amt in del_usets.keys():
                    del_vars[pfx_code][mark][del_amt] = self.builder.install_uset(del_usets[del_amt])
                by_pos[pfx_code][mark] = {}
                for pos in sorted(bit_xfrm_sets.keys()):
                    by_pos[pfx_code][mark][pos] = {}
                    for bit in bit_xfrm_sets[pos].keys():
                        by_pos[pfx_code][mark][pos][bit] = self.builder.install_uset(bit_xfrm_sets[pos][bit])
            s += self.builder.generate_scope_compilations()

            for mark in sorted(pass_data[pfx_code].keys()):
                mark_code = 1 # default
                if pass_no in self.conditional_mark_codes.keys():
                    if pfx_code in self.conditional_mark_codes[pass_no].keys():
                        mark_code = self.conditional_mark_codes[pass_no][pfx_code][mark]
                s += "    PabloAST * foundMark_%x = bnc.EQ(markCode, %i);\n" % (mark, mark_code)
                for pos in sorted(by_pos[pfx_code][mark].keys()):
                    for bit in sorted(by_pos[pfx_code][mark][pos].keys()):
                        bit_xfrm = by_pos[pfx_code][mark][pos][bit]
                        bit_xfrm = "%s.createAnd(%s, %s)" % (scope, bit_xfrm, "foundMark_%x" % mark)
                        if (pos > 0):
                            bit_xfrm = "%s.createAdvance(%s, %i)" % (scope, bit_xfrm, pos)
                        s += "    xfrm_%s[%i] = %s.createOr(xfrm_%s[%i], %s);\n" % (code_str, bit, scope, code_str, bit, bit_xfrm)
                for del_amt in sorted(del_vars[pfx_code][mark].keys()):
                    s += "    del_%s = %s.createOr(del_%s, %s);\n" % (code_str, scope, code_str, del_vars[pfx_code][mark][del_amt])
                    for d in range(1, del_amt):
                        s += "    del_%s = %s.createOr(del_%s, %s.createAdvance(%s, %i));\n" % (code_str, scope, code_str, scope, del_vars[pfx_code][mark][del_amt], d)
            s += finalize_long_composable_pfx_code(pfx_code)
            s += self.builder.close_scope()
        s += singleton_final_code  # long_composable and singleton have common final code
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
            cp1_list = uset_to_member_list(rg_set_map[pfx_code])
            for cp1 in cp1_list:
                if not cp1 in generated.keys():
                    generated[cp1] = self.builder.install_uset(singleton_uset(cp1))
                for cp2 in self.short_composable_map[cp1].keys():
                    if not cp2 in generated.keys():
                        generated[cp2] = self.builder.install_uset(singleton_uset(cp2))
            s += self.builder.generate_scope_compilations()
            case_code = {}
            deferred = []
            for cp1 in cp1_list:
                for cp2 in self.short_composable_map[cp1].keys():
                    precomposed = self.short_composable_map[cp1][cp2]
                    case_code[(cp1, cp2)] = generate_short_case_code(code_str, cp1, cp2, generated, precomposed)
                    if precomposed in cp1_list:
                        deferred.append(precomposed)
            precomposed_generated = {}
            for cp1 in cp1_list:
                if not cp1 in deferred:
                    s += generate_short_case_starter_code(scope, cp1, generated)
                    for cp2 in self.short_composable_map[cp1].keys():
                        s += case_code[(cp1, cp2)]
                        precomp = self.short_composable_map[cp1][cp2]
                        found = "found_%x_%x" % (cp1, cp2)
                        precomposed_generated[precomp] = found
            while len(deferred) > 0:
                next_deferred = []
                for cp1 in deferred:
                    if cp1 in precomposed_generated.keys():
                        generated[cp1] = "%s.createOr(%s, %s)" % (scope, generated[cp1], precomposed_generated[cp1]) 
                        s += generate_short_case_starter_code(scope, cp1, generated)
                        for cp2 in self.short_composable_map[cp1].keys():
                            s += case_code[(cp1, cp2)]
                            precomp = self.short_composable_map[cp1][cp2]
                            found = "found_%x_%x" % (cp1, cp2)
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

    def emit_normalization_cpp(self):
        basename = 'normalization-generated'
        f = cformat.open_cpp_file_for_write(basename)
        kernels = ""
        kernels += self.generate_u8_insertion_kernel()
        kernels += self.generate_singleton_stage()
        kernels += self.generate_excluded_composite_stage()
        kernels += self.generate_short_composable_stage()
        for pass_no in range(5):
            kernels += self.generate_nfc_stage(pass_no)
            kernels += self.generate_long_composition_xfrm_stage(pass_no)
        uset_definitions = self.builder.get_uset_definitions()
        t = string.Template(nfc_generated_cpp_template)
        s = t.substitute(uset_declarations = uset_definitions, kernels = kernels)
        f.write(s)
        cformat.close_cpp_file(f)

if __name__ == "__main__":
    ucd = UCD_database()
    generator = NFC_generator(ucd)
    generator.create_mappings()
    generator.create_max_insert_map()
    generator.allocate_ccc_passes()
    generator.create_conditional_mark_codes()
    #generator.show_ccc_pass_allocation()
    generator.show_conditional_codes()
    #generator.display_singletons()
    #generator.display_non_starter_decomps()
    #generator.display_short_composables()
    #generator.display_long_composables()
    #generator.display_excluded_composites()
    #generator.display_recursive_decomposables()
    #generator.display_max_insert_map()
    generator.emit_normalization_cpp()
