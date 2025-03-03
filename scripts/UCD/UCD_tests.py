#
# UCD_test.py - parsing Unicode Character Database (UCD) files
# and generating test files
# representation.
#
# Robert D. Cameron
# January 31, 2025
#
# Licensed under Open Software License 3.0.
#
#
import string, os.path, shutil
from UCD_parser import *

QA_dir = UCD_config.UCD_output_dir + "/QA"

def generateQA():
    if os.path.exists(QA_dir + ".bak"):
        shutil.rmtree(QA_dir + ".bak")
    if os.path.exists(QA_dir):
        shutil.move(QA_dir, QA_dir + ".bak")
    os.mkdir(QA_dir)

def generateCodepointDataFile(fname, entries):
    f = open(QA_dir + '/' + fname, 'w')
    for e in entries:
        e+= "\n"
        f.write(e)
    f.close()

def generateNormalizationTestFiles():
    (source, NFC, NFD, NFKC, NFKD) = parse_NormalizationTest_txt()
    generateQA()
    generateCodepointDataFile('NF-source', source)
    generateCodepointDataFile('NFC', NFC)
    generateCodepointDataFile('NFD', NFD)
    generateCodepointDataFile('NFKC', NFKC)
    generateCodepointDataFile('NFKD', NFKD)

if __name__ == "__main__":
    generateNormalizationTestFiles()
