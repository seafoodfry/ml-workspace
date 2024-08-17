/*
    g++ -O3 -march=native cpu_features.cpp -o cpu_features
*/
#include <iostream>
#include <bitset>
#include <array>
#include <string>
#include <vector>
#include <cstring>
#include <cpuid.h>

class InstructionSet {
    class InstructionSetInternal {
    public:
        InstructionSetInternal() : 
            nIds_{ 0 }, 
            nExIds_{ 0 },
            isIntel_{ false },
            isAMD_{ false },
            f_1_ECX_{ 0 },
            f_1_EDX_{ 0 },
            f_7_EBX_{ 0 },
            f_7_ECX_{ 0 },
            f_81_ECX_{ 0 },
            f_81_EDX_{ 0 }
        {
            std::array<unsigned int, 4> cpui;

            // Calling __get_cpuid with 0x0 as the function_id argument
            // gets the number of the highest valid function ID.
            // See
            // https://github.com/gcc-mirror/gcc/blob/9fbbad9b6c6e7fa7eaf37552173f5b8b2958976b/gcc/config/i386/cpuid.h#L313-L319
            __get_cpuid(0, &cpui[0], &cpui[1], &cpui[2], &cpui[3]);
            nIds_ = cpui[0];

            for (unsigned int i = 0; i <= nIds_; ++i) {
                //            leaf, subleaf, EAX,      EBX,      ECX,     EDX.
                __get_cpuid_count(i, 0, &cpui[0], &cpui[1], &cpui[2], &cpui[3]);
                data_.push_back(cpui);
            }

            // Capture vendor string.
            // The reinterpret_cast<int*>(vendor) is casting the address of the char array to an int pointer.
            // The * before the cast is dereferencing this pointer, allowing us to write directly to that memory location.
            // We're not converting to ASCII; instead, we're directly writing the 32-bit integer values from the
            // registers into the char array.
            // In x86 processors, the vendor string is stored as a sequence of ASCII characters in the EBX, EDX, and ECX
            // registers (in that order). Each register holds 4 bytes (32 bits), which corresponds to 4 characters.
            // reinterpret_cast is a C++ casting operator used for low-level reinterpretation of bit patterns.
            // It's treating the char array vendor as if it were an int*, allowing us to write a 32-bit integer
            // directly into the character array. This is a bit-level operation used here to efficiently copy
            // the vendor string from the CPUID result.
            // The contents of the registers are 4 bytes each and the null pointer takes an additional seat in
            // the array.
            char vendor[0x20];
            memset(vendor, 0, sizeof(vendor));
            *reinterpret_cast<int*>(vendor) = data_[0][1];
            *reinterpret_cast<int*>(vendor + 4) = data_[0][3];
            *reinterpret_cast<int*>(vendor + 8) = data_[0][2];
            vendor_ = vendor;
            // The followin comparisons work because of C strings are null terminated and the code stop looking after
            // the null terminator.
            // The null terminator comes when we do the memset because 0 == '\0'.
            // Then when we copy the contents to vendor_, vendor_ only gets up to the first '\0'.
            if (vendor_ == "GenuineIntel") {
                isIntel_ = true;
            } else if (vendor_ == "AuthenticAMD") {
                isAMD_ = true;
            }

            // load bitset with flags for function 0x00000001
            if (nIds_ >= 1)
            {
                f_1_ECX_ = data_[1][2];
                f_1_EDX_ = data_[1][3];
            }

            // load bitset with flags for function 0x00000007
            if (nIds_ >= 7)
            {
                f_7_EBX_ = data_[7][1];
                f_7_ECX_ = data_[7][2];
            }

            // Calling __get_cpuid with 0x80000000 as the function_id argument
            // gets the number of the highest valid extended ID.
            __get_cpuid(0x80000000, &cpui[0], &cpui[1], &cpui[2], &cpui[3]);
            nExIds_ = cpui[0];

            char brand[0x40];
            memset(brand, 0, sizeof(brand));

            for (unsigned int i = 0x80000000; i <= nExIds_; ++i)
            {
                __get_cpuid_count(i, 0, &cpui[0], &cpui[1], &cpui[2], &cpui[3]);
                extdata_.push_back(cpui);
            }

            // load bitset with flags for function 0x80000001
            if (nExIds_ >= 0x80000001)
            {
                f_81_ECX_ = extdata_[1][2];
                f_81_EDX_ = extdata_[1][3];
            }

            // Interpret CPU brand string if reported
            if (nExIds_ >= 0x80000004)
            {
                memcpy(brand, extdata_[2].data(), sizeof(cpui));
                memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
                memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
                brand_ = brand;
            }
        };

        unsigned int nIds_;
        unsigned int nExIds_;
        std::string vendor_;
        std::string brand_;
        bool isIntel_;
        bool isAMD_;
        std::bitset<32> f_1_ECX_;
        std::bitset<32> f_1_EDX_;
        std::bitset<32> f_7_EBX_;
        std::bitset<32> f_7_ECX_;
        std::bitset<32> f_81_ECX_;
        std::bitset<32> f_81_EDX_;
        std::vector<std::array<unsigned int, 4>> data_;
        std::vector<std::array<unsigned int, 4>> extdata_;
    };

public:
    static const InstructionSetInternal& CPU_Rep() { static InstructionSetInternal cpu_rep; return cpu_rep; }

    static bool SSE2() { return CPU_Rep().f_1_EDX_[26]; }
    static bool AVX() { return CPU_Rep().f_1_ECX_[28]; }
    static bool AVX2() { return CPU_Rep().f_7_EBX_[5]; }
    static bool AVX512F() { return CPU_Rep().f_7_EBX_[16]; }
};

bool check_sse2() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return edx & bit_SSE2;
}

bool check_avx() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return ecx & bit_AVX;
}


int main()
{
    std::cout << "CPU Vendor: " << InstructionSet::CPU_Rep().vendor_ << std::endl;
    std::cout << "CPU Brand: " << InstructionSet::CPU_Rep().brand_ << std::endl;
    std::cout << "SSE2 support: " << (InstructionSet::SSE2() ? "Yes" : "No") << std::endl;
    std::cout << "AVX support: " << (InstructionSet::AVX() ? "Yes" : "No") << std::endl;
    std::cout << "AVX2 support: " << (InstructionSet::AVX2() ? "Yes" : "No") << std::endl;
    std::cout << "AVX-512 support: " << (InstructionSet::AVX512F() ? "Yes" : "No") << std::endl;

    std::cout << "\nSimple checks...\n";
    std::cout << "SSE2 support: " << (check_sse2() ? "Yes" : "No") << std::endl;
    std::cout << "AVX support: " << (check_avx() ? "Yes" : "No") << std::endl;

    return 0;
}