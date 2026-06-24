
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>

#include <fcntl.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/raw_ostream.h>
#include "llvm/Support/CommandLine.h"

#include <kernel/basis/s2p_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/driver/driver.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/unicode/UCD_property_kernel.h>
#include <kernel/util/hex_convert.h>

#include <re/adt/re_re.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/unicode/resolve_properties.h>

#include <kernel/pipeline/program_builder.h>

namespace ustats
{
    namespace properties
    {
        std::string s_prop;
        std::ofstream fileOutputStream;

        extern "C" {
            void countOccurences( uint32_t occurences )
            {
                std::string output = " | " + std::to_string(occurences);

                if(fileOutputStream.is_open())
                {
                    fileOutputStream << output;
                }
                else
                {
                    std::cout << output;
                }
            }
        }

        class OutputOccurencesKernel : public pablo::PabloKernel  
        {
        public:
            OutputOccurencesKernel(KernelBuilder & b, kernel::StreamSet* streamData)
                : PabloKernel( b, "OutputOccurences",
                { kernel::Binding{"streamData", streamData} },
                {},
                {},
                { kernel::Binding{b.getSizeTy(), "occurences"} }
                )
            {}

        protected:
            void generatePabloMethod() override
            {
                pablo::PabloBuilder pb(getEntryScope());
                std::unique_ptr<cc::CC_Compiler> ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(getEntryScope(), getInputStreamSet("streamData"));

                pablo::Var* occurences = getOutputScalarVar("occurences");
                pablo::PabloAST* oneByte = ccc->compileCC(re::makeByte(0x01));
                pb.createAssign(occurences, pb.createCount(oneByte));
            }
        };
        
        typedef void (*UStatsPropertyType)( uint32_t fd );

        UStatsPropertyType generatePipeline(CPUDriver & driver, std::string property)
        {
            auto P = CreatePipeline(driver, kernel::Input<uint32_t>{"inputFileDescriptor"});

            kernel::Scalar* fileDescriptor = P.getInputScalar("inputFileDescriptor");
            kernel::StreamSet* byteStream = P.CreateStreamSet(1, 8);
            P.CreateKernelCall<kernel::ReadSourceKernel>(
                fileDescriptor,
                byteStream
            );

            kernel::StreamSet * BasisBits = P.CreateStreamSet(8);
            P.CreateKernelCall<kernel::S2PKernel>(byteStream, BasisBits);

            kernel::StreamSet* outputStream = P.CreateStreamSet(1);

            re::PropertyExpression * wordProp = re::makePropertyExpression(re::PropertyExpression::Kind::Codepoint, property);
            wordProp = cast<re::PropertyExpression>(UCD::linkAndResolve(wordProp));

            P.CreateKernelFamilyCall<kernel::UnicodePropertyKernelBuilder>(
                wordProp,
                BasisBits,
                outputStream
            );

            kernel::Kernel* occurenceKernel = P.CreateKernelCall<OutputOccurencesKernel>(outputStream);
            kernel::Scalar* numberOfOccurences = occurenceKernel->getOutputScalarAt(0);

            // P.CreateCall("countOccurences", countOccurences, {numberOfOccurences, outputFileDescriptor});
            P.CreateCall("countOccurences", countOccurences, {numberOfOccurences});

            return P.compile();
        }

        // todo: integrate with llvm
        int ParseProperty( std::string property, std::vector<std::string> files, std::string outputFile )
        {
            s_prop = property;

            CPUDriver driver("UStatsProperty");

            UStatsPropertyType parserFn = generatePipeline(driver, property);

            fileOutputStream.open(outputFile);
            if (fileOutputStream.is_open())
            {
                fileOutputStream << "| " << property;
            }
            else
            {
                std::cout << "| " << property;
            }

            for(auto& fileItr : files)
            {
                const int fd = open(fileItr.c_str(), O_RDONLY);
                if (LLVM_UNLIKELY(fd == -1)) {
                    llvm::errs() << "Error: cannot open " << fileItr << " for processing. Skipped.\n";
                } else {
                    parserFn(fd);
                    close(fd);
                }
            }

            if (fileOutputStream.is_open())
            {
                fileOutputStream << " |" << std::endl << std::endl;
            }
            else
            {
                std::cout << " |" << std::endl << std::endl;
            }

            fileOutputStream.close();

            return 0;
        }
    }
}

enum DisplayMode {
    CharacterDisplay, Codepoint, Name
};

enum OutputType {
    txt, csv
};

llvm::cl::list<std::string> InputFilenames(llvm::cl::Positional, llvm::cl::desc("<input file1> <input file2> ... <input fileN>"), llvm::cl::OneOrMore);
//llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"));
llvm::cl::opt<std::string> UnicodeProperty("prop", llvm::cl::desc("Specify unicode property"), llvm::cl::value_desc("property"));
llvm::cl::opt<std::string> CharacterClass("cc", llvm::cl::desc("Specify character class(es)"), llvm::cl::value_desc("ccregex"));
llvm::cl::opt<OutputType> OutputFormat("format", llvm::cl::desc("Specify output format"),
    llvm::cl::values(
        clEnumVal(txt, "Print output in text-based table"),
        clEnumVal(csv, "Print output in comma separated value table")));
llvm::cl::opt<DisplayMode> CharDisplay("char-display", llvm::cl::desc("Specify character display"), 
    llvm::cl::values(
        clEnumValN(CharacterDisplay, "default", "Display the default character"),
        clEnumVal(Codepoint, "Display in hexadecimal codepoint"),
        clEnumVal(Name, "Display name of unicode character")));

int main(int argc, char** argv)
{
    llvm::cl::ParseCommandLineOptions(argc, argv);
    std::ofstream Output;
    std::ifstream Input;
    std::vector<std::string> it = InputFilenames;
    std::vector<int> var_count(it.size(), 0);
    int i = 0;
    int j = 0;
    
    std::cout<<std::endl;//for python script test cases

    if (OutputFormat) {
        std::cout << "Output formatting not implemented yet" << std::endl;
    }

    if (CharDisplay) {
        std::cout << "Character display formatting not implemented yet" << std::endl;
    }

    if (strcmp(UnicodeProperty.c_str(),"") && strcmp(CharacterClass.c_str(),"")) {
        std::cout << "Property and character class functions are mutually exclusive" << std::endl;
    } else if ((strcmp(UnicodeProperty.c_str(),"") == 0) && (strcmp(CharacterClass.c_str(),"") == 0)) {
        std::cout << "Must specify property or character class" << std::endl;
    } else if (strcmp(UnicodeProperty.c_str(),"") != 0) {
        ustats::properties::ParseProperty( UnicodeProperty.c_str(), InputFilenames, "" );
    } else if (strcmp(CharacterClass.c_str(),"") != 0) {
        for(auto& fileItr : it){
            Input.open(fileItr.c_str());

            if ( Input.is_open() ) {
            char mychar;
            int stringlength = CharacterClass.length();
            char cc_array[stringlength + 1];
            strcpy(cc_array, CharacterClass.c_str());
                while ( Input ) {
                    mychar = Input.get();
                    //may in future include options such as [a-z], etc...
                    if(mychar == cc_array[0])
                        var_count[i] += 1;
                }
            }

            Input.close();


            std::cout << "File: " << fileItr.c_str() << std::endl;
            std::cout <<"CharacterClass: | " << CharacterClass.c_str() << " | " << var_count[i] <<" |"<< std::endl;
            i++;
        }
        std::cout<<std::endl;//for python script test cases
        //std::cout<<"test"<<std::endl;//we can do make ustats instead of doing make every time
        //Output is seperated in case we wish to include options to change output format
        //Output.open("output.txt", std::ios::out | std::ios::app);
        // Output.open("output.txt");
        // if(Output.is_open()){
        //     for(auto& fileItr : it){
        //         Output << "File: " << fileItr.c_str() << std::endl;
        //         Output <<"CharacterClass: | " << CharacterClass.c_str() << " | " << var_count[j] <<" |"<< std::endl;
        //         j++;
        //     }    
        // }
        // Output.close();
    } else {
        std::cout << "Something has gone terribly wrong..." << std::endl;
    }
}
