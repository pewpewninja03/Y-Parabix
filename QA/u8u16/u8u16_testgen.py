from random import randint
from binascii import b2a_hex

def random_codepoint(UTF8_lgth):
    if UTF8_lgth == 1: return randint(0, 0x7F)
    if UTF8_lgth == 2: return randint(0, 0x7FF)
    if UTF8_lgth == 3: codepoint = randint(0, 0xF7FF)
    else: codepoint = randint(0, 0x10F7FF)
    if codepoint <= 0xD7FF: return codepoint
    else: return codepoint + 0x800

def UTF8_length(cp):
    if cp <= 0x7F: return 1
    elif cp <= 0x7FF: return 2
    elif cp <= 0xFFFF: return 3
    else: return 4

def random_sequence_of_given_UTF8_length(pfx_type, lgth):
    seq = []
    remaining = lgth
    while remaining > 0:
        cp = random_codepoint(pfx_type)
        cp_lgth = UTF8_length(cp)
        if cp_lgth <= remaining:
            seq.append(cp)
            remaining -= cp_lgth
    return seq

def codepoint_to_UTF8(cp):
    if cp <= 0x7F: return [cp]
    suffix = 0x80 + (cp & 0x3F)
    if cp <= 0x7FF:
        return [0xC0 + (cp >> 6), suffix]
    suffix2 = 0x80 + ((cp >> 6) & 0x3F)
    if cp <= 0xFFFF:
        return [0xE0 + (cp >> 12), suffix2, suffix]
    suffix3 = 0x80 + ((cp >> 12) & 0x3F)
    return [0xF0 + (cp >> 18), suffix3, suffix2, suffix]
    
def codepoint_to_UTF16BE(cp):
    if cp <= 0xFFFF:
        return [cp >> 8, cp & 0xFF]
    else:
        b = cp - 0x10000
        return [0xD8 + (b >> 18), (b >> 10) & 0xFF, 0xDC + ((b >> 8) & 0x3), b & 0xFF]

def codepoint_seq_to_UTF8(cpseq):
    utf8 = []
    for cp in cpseq: utf8 += codepoint_to_UTF8(cp)
    return utf8

def codepoint_seq_to_UTF16BE(cpseq):
    utf16 = []
    for cp in cpseq: utf16 += codepoint_to_UTF16BE(cp)
    return utf16

def random_nonsuffix():
    byte = randint(0, 0xFF)
    if byte <= 0x7F: return byte
    else: return randint(0xC2, 0xF4)

def gen_UTF8_error_sequences():
    return [[randint(0xC0, 0xC1), randint(0x80, 0xBF)],
             [randint(0xF5, 0xFF), randint(0x80, 0xBF), randint(0x80, 0xBF), randint(0x80, 0xBF)],
             [0xE0, randint(0x80, 0x9F), randint(0x80, 0xBF)],
             [0xED, randint(0xA0, 0xBF), randint(0x80, 0xBF)],
             [0xF0, randint(0x80, 0x8F), randint(0x80, 0xBF), randint(0x80, 0xBF)],
             [0xF4, randint(0x90, 0xBF), randint(0x80, 0xBF), randint(0x80, 0xBF)],
             [randint(0xC2, 0xDF), random_nonsuffix()],
             [randint(0xE0, 0xEF), random_nonsuffix(), randint(0x80, 0xBF)],
             [randint(0xE1, 0xEF), randint(0x80, 0x9F), random_nonsuffix()],
             [randint(0xF0, 0xF4), random_nonsuffix(), randint(0x80, 0xBF), randint(0x80, 0xBF)],
             [randint(0xF0, 0xF3), randint(0x90, 0xBF), random_nonsuffix(),  randint(0x80, 0xBF)],
             [randint(0xF1, 0xF4), randint(0x80, 0x8F), randint(0x80, 0xBF), random_nonsuffix()],
             [randint(0x80, 0xBF)]]
             
def gen_UTF8_incomplete_sequences():
    return [[randint(0xC2, 0xDF)], [randint(0xE0, 0xEF)], [randint(0xF0, 0xF4)], 
             [randint(0xE0, 0xEC), randint(0xA0, 0xBF)],
             [randint(0xE1, 0xEF), randint(0x80, 0x9F)],
             [randint(0xF0, 0xF3), randint(0x90, 0xBF)],
             [randint(0xF1, 0xF4), randint(0x80, 0x8F)],
             [randint(0xF0, 0xF3), randint(0x90, 0xBF), randint(0x80, 0xBF)],
             [randint(0xF1, 0xF4), randint(0x80, 0x8F), randint(0x80, 0xBF)]]

def gen_err_string(error_sequence):
    s = ""
    for b in error_sequence: s += "%X" % b
    return s

prefix_groups = [(0,0), (1,4), (29,31), (60,64), (124,128), (129, 2045), (2046, 2047), (2048, 4196)]
suffix_groups = [(0,0), (1, 4), (5,128), (500, 5000)]

def generate_illegal_sequence_tests():
   for pfx in prefix_groups:
       for sfx in suffix_groups:
           for error_seq in gen_UTF8_error_sequences():
               pfx_lgth = randint(pfx[0], pfx[1])
               sfx_lgth = randint(sfx[0], sfx[1])
               pfx_type = randint(1,4)
               sfx_type = randint(1,4)
               pfx_seq = random_sequence_of_given_UTF8_length(pfx_type, pfx_lgth)
               sfx_seq = random_sequence_of_given_UTF8_length(sfx_type, sfx_lgth)
               err_string = gen_err_string(error_seq)
               filename = 'Illegal_UTF-8_%s@%i' % (err_string, pfx_lgth)
               f1 = open('TestFiles/' + filename, 'wb')
               for cp in pfx_seq: f1.write(bytes(codepoint_to_UTF8(cp)))
               f1.write(bytes(error_seq))
               for cp in sfx_seq: f1.write(bytes(codepoint_to_UTF8(cp)))
               f1.close()
               f2 = open('ExpectedOutput/Files/' + filename, 'wb')
               for cp in pfx_seq: f2.write(bytes(codepoint_to_UTF16BE(cp)))
               f2.close()
               f3 = open('ExpectedOutput/Messages/' + filename, 'w')
               f3.write("Illegal UTF-8 sequence at position %i in source.\n" % pfx_lgth)
               f3.close()

def generate_incomplete_sequence_tests():
   for pfx in prefix_groups:
       for error_seq in gen_UTF8_incomplete_sequences():
           pfx_lgth = randint(pfx[0], pfx[1])
           pfx_type = randint(1,4)
           pfx_seq = random_sequence_of_given_UTF8_length(pfx_type, pfx_lgth)
           err_string = gen_err_string(error_seq)
           filename = 'Incomplete_UTF-8_%s@%i' % (err_string, pfx_lgth)
           f1 = open('TestFiles/' + filename, 'wb')
           for cp in pfx_seq: f1.write(bytes(codepoint_to_UTF8(cp)))
           f1.write(bytes(error_seq))
           f1.close()
           f2 = open('ExpectedOutput/Files/' + filename, 'wb')
           for cp in pfx_seq: f2.write(bytes(codepoint_to_UTF16BE(cp)))
           f2.close()
           f3 = open('ExpectedOutput/Messages/' + filename, 'w')
           f3.write("EOF with incomplete UTF-8 sequence at position %i in source.\n" % pfx_lgth)
           f3.close()


def gen_random_all_UTF8():
    f1 = open('TestFiles/All_Codepoint_UTF-8', 'wb')
    f2 = open('ExpectedOutput/Files/All_Codepoint_UTF-8', 'wb')
    f3 = open('ExpectedOutput/Messages/All_Codepoint_UTF-8', 'wb')
    tbl = {}
    for i in range(0, 0x20FFFF):
        cp = random_codepoint(4)
        tbl[cp] = 1
        f1.write(bytes(codepoint_to_UTF8(cp)))
        f2.write(bytes(codepoint_to_UTF16BE(cp)))
    for max_lgth in range(1,3):
        for i in range(0,4096):
            cp = random_codepoint(max_lgth)
            tbl[cp] = 1
            f1.write(bytes(codepoint_to_UTF8(cp)))
            f2.write(bytes(codepoint_to_UTF16BE(cp)))
    for i in range(0, 0xD7FF):
        if i not in tbl:
            f1.write(bytes(codepoint_to_UTF8(i)))
            f2.write(bytes(codepoint_to_UTF16BE(i)))
    for i in range(0xE000, 0x10FFFF):
        if i not in tbl:
            f1.write(bytes(codepoint_to_UTF8(i)))
            f2.write(bytes(codepoint_to_UTF16BE(i)))
    f1.close()
    f2.close()
    f3.close()
