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

class InstructionSet
{
    class InstructionSetInternal
    {
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
            __get_cpuid(0, &cpui[0], &cpui[1], &cpui[2], &cpui[3]);
            nIds_ = cpui[0];

            for (unsigned int i = 0; i <= nIds_; ++i)
            {
                __get_cpuid_count(i, 0, &cpui[0], &cpui[1], &cpui[2], &cpui[3]);
                data_.push_back(cpui);
            }

            // Capture vendor string
            char vendor[0x20];
            memset(vendor, 0, sizeof(vendor));
            *reinterpret_cast<int*>(vendor) = data_[0][1];
            *reinterpret_cast<int*>(vendor + 4) = data_[0][3];
            *reinterpret_cast<int*>(vendor + 8) = data_[0][2];
            vendor_ = vendor;
            if (vendor_ == "GenuineIntel")
            {
                isIntel_ = true;
            }
            else if (vendor_ == "AuthenticAMD")
            {
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

int main()
{
    std::cout << "CPU Vendor: " << InstructionSet::CPU_Rep().vendor_ << std::endl;
    std::cout << "CPU Brand: " << InstructionSet::CPU_Rep().brand_ << std::endl;
    std::cout << "SSE2 support: " << (InstructionSet::SSE2() ? "Yes" : "No") << std::endl;
    std::cout << "AVX support: " << (InstructionSet::AVX() ? "Yes" : "No") << std::endl;
    std::cout << "AVX2 support: " << (InstructionSet::AVX2() ? "Yes" : "No") << std::endl;
    std::cout << "AVX-512 support: " << (InstructionSet::AVX512F() ? "Yes" : "No") << std::endl;

    return 0;
}