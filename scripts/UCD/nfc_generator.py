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

class Uset_Builder:
    def __init__(self):
        self.installed_usets = {}
        self.role_to_key_map = {}
        self.key_to_role_map = {}
        self.current_scope_roles = []

    def open_scope(self, scope):
        self.current_scope = scope
        self.current_scope_roles = []

    def install_uset(self, uset):
        key = uset_structural_key(uset)
        if not key in self.installed_usets.keys():
            #print("new key: %s" % key)
            self.installed_usets[key] = uset
            self.current_scope_roles.append(key)
        return key

    def install_uset_for_role(self, role_name, uset):
        key = uset_structural_key(uset)
        self.role_to_key_map[role_name] = key
        if not key in self.installed_usets.keys():
            #print("new key: %s" % key)
            self.installed_usets[key] = uset
            self.key_to_role_map[key] = [role_name]
        else:
            self.key_to_role_map[key].append(role_name)
        self.current_scope_roles.append(role_name)

    def install_uset_family(self, role_template, usets):
        for i in range(len(usets)):
            self.install_uset_for_role(role_template % i, usets[i])

    def get_uset_key_for_role(self, role_name):
        if not role_name in self.role_to_key_map.keys():
            raise Exception("Unknown role name: " + role_name)
        return self.role_to_key_map[role_name]

    def get_installed_usets(self):
        return self.installed_usets

    def get_role_to_key_map(self):
        return self.role_to_key_map

    def generate_scope_compilations(self, movement_mode = "LookAhead"):
        pfx = self.current_scope
        if movement_mode != "LookAhead":
            pfx = pfx + "_adv"
        t1 = string.Template(scoped_compiler_template)
        t2 = string.Template(assignment_template)
        assigs = ""
        for i in range(len(self.current_scope_roles)):
            assigs += t2.substitute(pfx = pfx, scope = self.current_scope, index = i, role = self.current_scope_roles[i])
        defs = t1.substitute(pfx = pfx,
                             scope = self.current_scope,
                             movement = movement_mode,
                             num_roles = len(self.current_scope_roles),
                             assignments = assigs)
        return defs

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

class U8_Translation_Generator:
    def __init__(self):
        self.codepoint_maps = {}
        self.builder = Uset_Builder()

    def define_codepoint_map(self, skey, map):
        self.codepoint_maps[skey] = map

    def has_codepoint_map(self, skey):
        return skey in self.codepoint_maps.keys()

    def generate_basis_bit_translation_usets(self, skey):
        if not self.has_codepoint_map(skey):
            raise Exception("Requesting basis bit translations for %s no such codepoint_map" % (skey))
        translation_map = self.codepoint_maps[skey]
        translation_sets = []
        for cp1 in translation_map.keys():
            cp2 = translation_map[cp1]
            bit_diffs = cp1 ^ cp2
            bit = 0
            while bit_diffs > 0:
                if len(translation_sets) <= bit:
                    translation_sets.append(empty_uset())
                if bit_diffs & 1 == 1:
                    translation_sets[bit] = uset_union(translation_sets[bit], singleton_uset(cp1))
                bit_diffs >>=  1
                bit += 1
        self.builder.install_uset_family(skey + "_basis_%02i", translation_sets)

    def generate_u8_adjustment_bixnum_usets(self, skey):
        if not self.has_codepoint_map(skey):
            raise Exception("Requesting insertion bixnums for %s no such codepoint_map" % (skey))
        translation_map = self.codepoint_maps[skey]
        (insertion_bixnum_usets, deletion_bixnum_usets) = u8_adjustment_bixnums_from_codepoint_map(translation_map)
        if len(insertion_bixnum_usets) > 0:
            self.builder.install_uset_family(skey + "_insert_basis_%i", insertion_bixnum_usets)
        if len(deletion_bixnum_usets) > 0:
            self.builder.install_uset_family(skey + "_delete_basis_%i", deletion_bixnum_usets)

    def print_uset_definitions(self):
        print(self.builder.generate_uset_definitions("uset_%i"))

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
class SingletonCanonicalization : public PabloKernel {
public:
    SingletonCanonicalization
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * SelectMask, StreamSet * XfrmBasis);
protected:
    void generatePabloMethod() override;
};

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

singleton_pfx_template = r"""
    auto ${builder} = pb.createScope();
    PabloAST * ${pfx_test_var} = ${logic};
    pb.createIf(${pfx_test_var}, ${builder});
    std::vector<PabloAST *> xfrm_${code_str}(8, All0);
    PabloAST * del_${code_str} = All0;

"""

def gen_singleton_pfx_code(pfx_code):
    code_str = pfx_code_string(pfx_code)
    pfx_test_var = "pfx_%s_test" % code_str
    test = prefix_test_logic(pfx_code)
    scope = "b_%s" % code_str
    t = string.Template(singleton_pfx_template)
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
    xlate_code = CharacterTranslationLogic(cp, canon, Num0s, cp_var, "xfrm_%s" % code_str, "b_%s" % code_str)
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

singleton_pfx_final_template = r"""
    for (unsigned i = 0; i < 8; i++) {
        ${builder}.createAssign(XfrmVar[i], $builder.createOr(XfrmVar[i], xfrm_${code_str}[i]));
    }
"""

update_del_var_template = r"""
    ${builder}.createAssign(DeleteVar, $builder.createOr(DeleteVar, del_${code_str}));
"""

def finalize_singleton_pfx_code(pfx_code, has_del):
    code_str = pfx_code_string(pfx_code)
    scope = "b_%s" % code_str
    t1 = string.Template(singleton_pfx_final_template)
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

pass_template = r"""//
class FindComposables${pass_no} : public PabloKernel {
public:
    FindComposables${pass_no}
        (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * ccc_NR${extra_inputs},
                                       StreamSet * MarksFound, StreamSet * Index_ccc_NR_or_MarksFound);
protected:
    void generatePabloMethod() override;
};

FindComposables${pass_no}::FindComposables${pass_no}
    (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * ccc_NR${extra_inputs},
                                   StreamSet * MarksFound, StreamSet * Index_ccc_NR_or_MarksFound) :
: PabloKernel(ts, "FindComposables${pass_no}" + Basis->shapeString(),
{Binding{"Basis", Basis}, Binding{"ccc_NR", ccc_NR}${extra_bindings}},
{Binding{"MarksFound", MarksFound}, Binding{"Index_ccc_NR_or_MarksFound", Index_ccc_NR_or_MarksFound}}) {}

void FindComposables${pass_no}::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    PabloAST * ccc_NR = getInputStreamSet("ccc_NR")[0];
    PabloAST * eligible_starter = ${eligible};
    Var * composable2nd = pb.createVar("composable2nd", All0);
"""

def gen_pass_header_code(pass_no):
    s = string.Template(pass_template)
    extra_inputs = ""
    extra_bindings = ""
    eligible = "pb.createOnes()"
    if (pass_no > 0):
        extra_inputs = ", StreamSet * priorFound"
        extra_bindings = r""", Binding{"priorFound", priorFound}"""
        eligible = r"""pb.createNot(getInputStreamSet("priorFound")[0])"""
    return s.substitute(pass_no = pass_no,
                        extra_inputs = extra_inputs,
                        extra_bindings = extra_bindings,
                        eligible = eligible)

pfx_template = r"""
    auto ${builder} = pb.createScope();
    PabloAST * ${pfx_test_var} = ${logic};
    pb.createIf(pb.createAnd(${pfx_test_var}, eligible_starter), ${builder});
"""

def gen_pass_pfx_code(pfx_code):
    code_str = pfx_code_string(pfx_code)
    pfx_test_var = "pfx_%s_test" % code_str
    test = prefix_test_logic(pfx_code)
    if pfx_code == 0:
        test = "pb.createAnd(%s, pb.createLookAhead(ccc_NR, 1))" % test
    scope = "b_%s" % code_str
    t = string.Template(pfx_template)
    return t.substitute(builder = scope,
                        pfx_test_var = pfx_test_var,
                        logic = test)


mark_case_template = r"""
//  Case for mark ${mark}
    PabloAST * possible_${mark}_pos = ${builder}.createScanTo(${mark_starters}, ${ccc_enum}_or_NR);
    PabloAST * found_${mark} = ${builder}.createAnd(possible_${mark}_pos, ${mark_var});
    ${builder}.createAssign(composable2nd, ${builder}.createOr(composable2nd, found_${mark}));
"""

def gen_mark_case_logic(mark, starters_var, mark_var, ccc_enum, code_str):
    t = string.Template(mark_case_template)
    return t.substitute(builder = "b_%s" % code_str,
                        mark="%x" % mark,
                        mark_var = mark_var,
                        mark_starters=starters_var,
                        ccc_enum=ccc_enum)

finalize_nfc_template = r"""// Generate combined outputs for pass ${pass_no}.
    Var * marksFoundVar = pb.getOutputStreamVar("MarksFound");
    pb.createAssign(pb.createExtract(marksFoundVar, pb.getInteger(0)), composable2nd);
    PabloAST * updatedIndexStrm = pb.createOr(ccc_NR, composable2nd);
    Var * indexVar = pb.getOutputStreamVar("Index_ccc_NR_or_MarksFound");
    pb.createAssign(pb.createExtract(indexVar, pb.getInteger(0)), updatedIndexStrm);
}
"""

def finalize_nfc_stage(pass_no):
    s = string.Template(finalize_nfc_template)
    return s.substitute(pass_no = pass_no)

short_composable_header = r"""//
class ShortComposableTranslation : public PabloKernel {
public:
    ShortComposableTranslation
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * DeletePrior, StreamSet * XfrmBasis);
protected:
    void generatePabloMethod() override;
};

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

def generate_short_case_code(code_str, cp1, cp2, generated, precomposed, Num0s):
    t = string.Template(short_composable_case_template)
    s = t.substitute(code_str = code_str,
                        cp1 = "%x" % cp1,
                        cp2 = "%x" % cp2,
                        cp1_var = generated[cp1],
                        cp2_var = generated[cp2],
                        cp1_len=u8_encoded_length(cp1),
                        precomposed = "%x" % precomposed)
    xlate_code = CharacterTranslationLogic(cp2, precomposed, Num0s, "found_%x_%x" % (cp1, cp2), "xfrm_%s" % code_str, "b_%s" % code_str)
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
#    - Num0s zero bytes have been inserted the first byte of cp1
#    - Num0s + u8_encoded_length(cp1) >= u8_encoded_length(cp2)
#    - excess = Num0s + u8_encoded_length(cp1) - u8_encoded_length(cp2)
#    - If excess > 0, the first excess positions are don't cares
#      and will ultimately be deleted
#    - pb is the PabloBuilder for logic
#
def CharacterTranslationLogic(cp1, cp2, Num0s, marker, basis_var, pb):
    s = ""
    len1 = u8_encoded_length(cp1)
    len2 = u8_encoded_length(cp2)
    excess = Num0s + len1 - len2
    for i in range(len2):
        # position for cp2 byte
        cp2_byte = u8_code_unit(cp2, i + 1)
        pos = excess + i
        cp1_byte = 0 # default for positions corresponding to inserted zeroes
        if pos == 0:
            cp1_byte = u8_code_unit(cp1, 1)
        elif pos > Num0s:
            cp1_byte = u8_code_unit(cp1, pos - Num0s + 1)
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

def u8_bit_transform_sets(translation_map, zero_insertion_map):
    bit_xfrm_sets = {}
    for cp1 in translation_map.keys():
        cp1_uset = singleton_uset(cp1)
        cp2 = translation_map[cp1]
        len1 = u8_encoded_length(cp1)
        len2 = u8_encoded_length(cp2)
        Num0s = 0
        if cp1 in zero_insertion_map.keys():
            Num0s = zero_insertion_map[cp1]
        excess = Num0s + len1 - len2
        for i in range(len2):
            # position for cp2 byte
            cp2_byte = u8_code_unit(cp2, i + 1)
            pos = excess + i # position relative to cp1 starter
            cp1_byte = 0 # default for positions corresponding to inserted zeroes
            if pos == 0:
                cp1_byte = u8_code_unit(cp1, 1)
            elif pos > Num0s:
                cp1_byte = u8_code_unit(cp1, pos - Num0s + 1)
            if cp1_byte != cp2_byte:
                diff = cp1_byte ^ cp2_byte
                if not pos in bit_xfrm_sets.keys():
                    bit_xfrm_sets[pos] = {}
                for j in range(8):
                    if (diff >> j) & 1 == 1:
                        if not j in bit_xfrm_sets[pos].keys():
                            bit_xfrm_sets[pos][j] = empty_uset()
                        bit_xfrm_sets[pos][j] = uset_union(bit_xfrm_sets[pos][j], cp1_uset)
    return bit_xfrm_sets

nonstarter_decomposition_template = r"""//
class NonStarterDecomposition : public PabloKernel {
public:
    NonStarterDecomposition
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * NSD_Basis);
protected:
    void generatePabloMethod() override;
};

NonStarterDecomposition::NonStarterDecomposition
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * NSD_Basis)
: PabloKernel(ts, "NonStarterDecomposition" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"NSD_Basis", NSD_Basis}}) {}

void NonStarterDecomposition::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    std::vector<Var *> NSD_Var(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        NSD_Var[i] = pb.createVar("NSD_Basis" + std::to_string(i), Basis[i]);
    }
${pablo_code}
    writeOutputStreamSet("NSD_Basis", NSD_Var);
}
"""

nonstarter_decomposition_case_template = r"""//  Case for ${cp}
    auto ${builder} = pb.createScope();
    pb.createIf(${cp_var}, ${builder});
    std::vector<PabloAST *> ${basis_var}(8, All0);
${cp1_logic}
    PabloAST * ins_${cp2} = ${builder}.createAdvance(${cp_var}, ${adv});
${cp2_logic}
    for (unsigned i = 0; i < 8; i++) {
        ${builder}.createAssign(NSD_Var[i], ${builder}.createXor(NSD_Var[i], ${basis_var}[i]));
    }
"""


def generate_nonstarter_decomposition_case(cp, cp_var, Num0s, cp1, cp2):
    t = string.Template(nonstarter_decomposition_case_template)
    builder = "b_%x" % cp
    basis_var = "xfrm_%x" % cp
    if Num0s != u8_encoded_length(cp2):
        raise Exception("Num0s != u8_encoded_length(cp2)")
    cp1_logic = CharacterTranslationLogic(cp, cp1, 0, cp_var, basis_var, builder)
    adv = u8_encoded_length(cp1)
    cp2_logic = CharacterInsertionLogic(cp2, "ins_%x" % cp2, basis_var, builder)
    s = t.substitute(builder = builder,
                        cp = "%x" % cp,
                        cp2 = "%x" % cp2,
                        cp_var = cp_var,
                        basis_var = basis_var,
                        adv = adv,
                        cp1_logic = cp1_logic,
                        cp2_logic = cp2_logic)
    return s

u8_insertion_bixnum_template = r"""//
class U8_InsertionBixNum : public PabloKernel {
public:
    U8_InsertionBixNum
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * InsertionBixNum);
protected:
    void generatePabloMethod() override;
};

U8_InsertionBixNum::U8_InsertionBixNum
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * InsertionBixNum)
: PabloKernel(ts, "U8_InsertionBixNum" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"InsertionBixNum", InsertionBixNum}}) {}

void U8_InsertionBixNum::generatePabloMethod() {
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


class NFC_generator(U8_Translation_Generator):
    def __init__(self, ucd):
        super().__init__()
        self.ucd = ucd
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

    def create_mappings(self):
        self.singleton_map = {}
        self.short_composable_map = {}
        self.long_composable_map = {}
        self.non_starter_decomposition_map = {}
        for precomposed in ucd.decomp_map.keys():
            if uset_member(ucd.dt_map['Can'], precomposed):
                decomp = ucd.decomp_map[precomposed]
                if len(decomp) == 1:
                    self.singleton_map[precomposed] = ord(decomp[0])
                elif len(decomp) == 2:
                    if uset_member(ucd.CE_map['Y'], precomposed):
                        # Composition exclusion - skip.
                        continue
                    cp1 = ord(decomp[0])
                    cp2 = ord(decomp[1])
                    if uset_member(ucd.ccc_map['NR'], precomposed):
                        if uset_member(ucd.ccc_map['NR'], cp1):
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
                            # non-starter decomposition case 1
                            self.non_starter_decomposition_map[precomposed] = (cp1, cp2)
                    else:
                        # non-starter decomposition case 2
                        self.non_starter_decomposition_map[precomposed] = (cp1, cp2)
                else:
                    raise Exception("Unexpected: decomposition length(%x) = %i" % (precomposed, len(decomp)))

    def update_max_insert_map(self, cp, insert_len):
        if cp in self.max_insert_map.keys():
            if insert_len > self.max_insert_map[cp]:
                self.max_insert_map[cp] = insert_len
        elif insert_len > 0:
            self.max_insert_map[cp] = insert_len

    def create_max_insert_map(self):
        self.max_insert_map = {}
        self.post_insert_map = {}
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
        for cp in self.non_starter_decomposition_map.keys():
            (cp1, cp2) = self.non_starter_decomposition_map[cp]
            if uset_member(ucd.dt_map['Can'], cp1):
                raise Exception("Unexpected further decomposition for %x" % cp1)
            if uset_member(ucd.dt_map['Can'], cp2):
                raise Exception("Unexpected further decomposition for %x" % cp2)
            ldiff = u8_encoded_length(cp1) - u8_encoded_length(cp)
            if ldiff > 0:
                self.update_max_insert_map(cp, ldiff)
            self.post_insert_map[cp] = u8_encoded_length(cp2)

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

    def cp_with_ccc(self, cp):
        return "%x(%s)" % (cp, self.ccc_val_map[cp])

    def display_singletons(self):
        for cp in sorted(self.singleton_map.keys()):
            canon_cp = self.singleton_map[cp]
            print("%s => %s" % (self.cp_with_ccc(cp), self.cp_with_ccc(canon_cp)))

    def display_non_starter_decomps(self):
        for cp in sorted(self.non_starter_decomposition_map.keys()):
            (cp1, cp2) = self.non_starter_decomposition_map[cp]
            print("%s => %s %s" % (self.cp_with_ccc(cp), self.cp_with_ccc(cp1), self.cp_with_ccc(cp2)))

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

    def generate_singleton_stage(self):
        s = singleton_header
        rg_set_map = range_usets_from_cps(self.singleton_map.keys())
        for pfx_code in rg_set_map.keys():
            code_str = pfx_code_string(pfx_code)
            scope = "b_%s" % code_str
            s += gen_singleton_pfx_code(pfx_code)
            self.builder.open_scope(scope)
            generated = {}
            singleton_list = uset_to_member_list(rg_set_map[pfx_code])
            for cp in singleton_list:
                generated[cp] = self.builder.install_uset(singleton_uset(cp))
            s += self.builder.generate_scope_compilations()
            DelTotal = 0
            for cp in singleton_list:
                canon = self.singleton_map[cp]
                Num0s = 0
                ToDel = 0
                if cp in self.max_insert_map.keys():
                    Num0s = self.max_insert_map[cp]
                cp_len = u8_encoded_length(cp)
                canon_len = u8_encoded_length(canon)
                if canon_len < cp_len:
                    ToDel = cp_len - canon_len
                    DelTotal += ToDel
                s += generate_singleton_code(code_str, cp, canon, generated[cp], Num0s, ToDel)
            s += finalize_singleton_pfx_code(pfx_code, DelTotal)
        s += singleton_final_code
        return s

    def generate_singleton_stage2(self):
        s = singleton_header
        rg_set_map = range_usets_from_cps(self.singleton_map.keys())
        for pfx_code in rg_set_map.keys():
            code_str = pfx_code_string(pfx_code)
            scope = "b_%s" % code_str
            s += gen_singleton_pfx_code(pfx_code)
            self.builder.open_scope(scope)
            pfx_xlate_map = {}
            singleton_list = uset_to_member_list(rg_set_map[pfx_code])
            for cp1 in singleton_list:
                pfx_xlate_map[cp1] = self.singleton_map[cp1]
            bit_xfrm_sets = u8_bit_transform_sets(pfx_xlate_map, self.max_insert_map)
            del_usets = u8_deletion_usets_from_codepoint_map(pfx_xlate_map)
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
            s += finalize_singleton_pfx_code(pfx_code, len(del_usets.keys()) > 0)
        s += singleton_final_code
        return s

    def generate_nfc_stage(self, pass_no):
        s = gen_pass_header_code(pass_no)
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
            self.builder.open_scope(scope)
            for mark in sorted(pass_data[pfx_code].keys()):
                starters_var[mark] = self.builder.install_uset(pass_data[pfx_code][mark])
                mark_var[mark] = self.builder.install_uset(singleton_uset(mark))
            s += self.builder.generate_scope_compilations()

            for mark in sorted(pass_data[pfx_code].keys()):
                ccc_enum = self.ccc_val_map[mark]
                s += gen_mark_case_logic(mark, starters_var[mark], mark_var[mark], ccc_enum, code_str)

        s+=finalize_nfc_stage(pass_no)
        return s

    def generate_short_composable_stage(self):
        s = short_composable_header
        rg_set_map = range_usets_from_cps(self.short_composable_map.keys())
        for pfx_code in rg_set_map.keys():
            code_str = pfx_code_string(pfx_code)
            scope = "b_%s" % code_str
            s += gen_short_composable_pfx_code(pfx_code)
            self.builder.open_scope(scope)
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
                    Num0s = 0
                    if cp2 in self.max_insert_map.keys():
                        Num0s = self.max_insert_map[cp2]
                    case_code[(cp1, cp2)] = generate_short_case_code(code_str, cp1, cp2, generated, precomposed, Num0s)
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
        s += short_composable_final_code
        return s

    def generate_u8_insertion_kernel(self):
        expansion_usets = insert_map_to_bixnum_usets(self.max_insert_map)
        insertion_usets = insert_map_to_bixnum_usets(self.post_insert_map)
        bixnum_bits = max(len(expansion_usets), len(insertion_usets))
        t = string.Template(u8_insertion_bixnum_template)
        s = t.substitute(insertion_bixnum_bits = bixnum_bits)
        self.builder.open_scope("pb")
        expansions = []
        for i in range(len(expansion_usets)):
            expansions.append(self.builder.install_uset(expansion_usets[i]))
        s += self.builder.generate_scope_compilations()
        self.builder.open_scope("pb")
        insertions = []
        for i in range(len(insertion_usets)):
            insertions.append(self.builder.install_uset(insertion_usets[i]))
        s += self.builder.generate_scope_compilations("Advance")
        for i in range(len(expansion_usets)):
            s += "    insertions[%s] = %s;\n" % (i, expansions[i])
        for i in range(len(insertion_usets)):
            if i >= len(expansion_usets):
                s += "    insertions[%s] = %s;\n" % (i, insertions[i])
            else:
                s += "    insertions[%s] = pb.createOr(insertions[%s], %s);\n" % (i, i, insertions[i])
        s += u8_insertion_final_code
        return s

    def generate_nonstarter_decompositions(self):
        self.builder.open_scope("pb")
        t = string.Template(nonstarter_decomposition_template)
        generated = {}
        for cp in self.non_starter_decomposition_map.keys():
            generated[cp] = self.builder.install_uset(singleton_uset(cp))
        code = self.builder.generate_scope_compilations()
        for cp in self.non_starter_decomposition_map.keys():
            (cp1, cp2) = self.non_starter_decomposition_map[cp]
            if uset_member(ucd.dt_map['Can'], cp1):
                raise Exception("Unexpected further decomposition for %x" % cp1)
            if uset_member(ucd.dt_map['Can'], cp2):
                raise Exception("Unexpected further decomposition for %x" % cp2)
            Num0s = self.post_insert_map[cp]
            cp_var = generated[cp]
            code += generate_nonstarter_decomposition_case(cp, cp_var, Num0s, cp1, cp2)
        s = t.substitute(pablo_code = code)
        return s


if __name__ == "__main__":
    ucd = UCD_database()
    generator = NFC_generator(ucd)
    generator.create_mappings()
    generator.create_max_insert_map()
    generator.allocate_ccc_passes()
    #generator.show_ccc_pass_allocation()
    #generator.display_singletons()
    #generator.display_non_starter_decomps()
    #generator.display_short_composables()
    #generator.display_long_composables()
    #generator.display_max_insert_map()
    kernels = ""
    kernels += generator.generate_singleton_stage2()
    #print(generator.generate_nfc_stage(0))
    #kernels += generator.generate_short_composable_stage()
    #generator.print_uset_definitions()
    #print(generator.generate_insertions_for_nonstarter_decompositions())
    #kernels += generator.generate_u8_insertion_kernel()
    #kernels += generator.generate_nonstarter_decompositions()
    generator.print_uset_definitions()
    print(kernels)
