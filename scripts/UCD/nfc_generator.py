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
    rg_strings = []
    for rg in rgs:
        (lo, hi) = rg
        if (lo == hi): rg_strings.append("%x" % lo)
        elif (lo + 1 == hi): rg_strings.append("%x_%x" % (lo, hi))
        else: rg_strings.append("%x___%x" % (lo, hi))
    return "uset_" + "_".join(rg_strings)

class Uset_Collector:
    def __init__(self):
        self.installed_usets = {}
        self.role_to_key_map = {}
        self.key_to_role_map = {}

    def install_uset_for_role(self, role_name, uset):
        key = uset_structural_key(uset)
        self.role_to_key_map[role_name] = key
        if not key in self.installed_usets.keys():
            self.installed_usets[key] = uset
            self.key_to_role_map[key] = [role_name]
        else:
            self.key_to_role_map[key].append(role_name)

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

    def generate_uset_definitions(self, name_template):
        defs = ""
        i = 0
        generated = {}
        for role in sorted(self.role_to_key_map.keys()):
            uset_key = self.role_to_key_map[role]
            if uset_key not in generated.keys():
                uset = self.installed_usets[uset_key]
                set_name = name_template % i
                defs += "    " + uset.generate(set_name)
                generated[uset_key] = set_name
                i += 1
            defs += "    const UnicodeSet & %s = %s;\n" % (role, generated[uset_key])
        return defs

class U8_Translation_Generator:
    def __init__(self):
        self.codepoint_maps = {}
        self.collector = Uset_Collector()

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
        self.collector.install_uset_family(skey + "_basis_%i", translation_sets)

    def generate_u8_adjustment_bixnum_usets(self, skey):
        if not self.has_codepoint_map(skey):
            raise Exception("Requesting insertion bixnums for %s no such codepoint_map" % (skey))
        translation_map = self.codepoint_maps[skey]
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
        if len(insertion_bixnum_usets) > 0:
            self.collector.install_uset_family(skey + "_insert_basis_%i", insertion_bixnum_usets)
        if len(deletion_bixnum_usets) > 0:
            self.collector.install_uset_family(skey + "_delete_basis_%i", deletion_bixnum_usets)

    def print_uset_definitions(self):
        print(self.collector.generate_uset_definitions("uset_%i"))


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
                                if not cp2 in self.short_composable_map.keys():
                                    self.short_composable_map[cp2] = {}
                                self.short_composable_map[cp2][cp1] = precomposed
                            else:
                                if not cp2 in self.long_composable_map.keys():
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
        by_starter = {}
        for cp2 in sorted(self.short_composable_map.keys()):
            for cp1 in self.short_composable_map[cp2].keys():
                cp = self.short_composable_map[cp2][cp1]
                if not cp1 in by_starter.keys():
                    by_starter[cp1] = {}
                by_starter[cp1][cp2] = cp
        for cp1 in sorted(by_starter.keys()):
            print("%s: " % self.cp_with_ccc(cp1))
            for cp2 in sorted(by_starter[cp1].keys()):
                cp = by_starter[cp1][cp2]
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

    def generate_singleton_code(self):
        self.define_codepoint_map("singletons", self.singleton_map)
        self.generate_basis_bit_translation_usets("singletons")
        self.generate_u8_adjustment_bixnum_usets("singletons")


if __name__ == "__main__":
    ucd = UCD_database()
    generator = NFC_generator(ucd)
    generator.create_mappings()
    generator.allocate_ccc_passes()
    generator.show_ccc_pass_allocation()
    #generator.display_singletons()
    #generator.display_non_starter_decomps()
    #generator.display_short_composables()
    #generator.display_long_composables()
    generator.generate_singleton_code()
    generator.print_uset_definitions()


