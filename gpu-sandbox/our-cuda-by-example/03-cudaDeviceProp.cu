/*
A g4dn.xlarge has the following output:

 --- General information for device 0 ---
	Name: Tesla T4
	Compute capability: 7.5
	Clock rate: 1590000
	Device copy overlap: enabled
	Kernel excition timeout: disabled
 --- General information for device ---
	Total globale memory: 15655829504
	Total constant memory: 65536
	Max memory pitch: 2147483647
	Texture alignment: 512
 --- MP information for device ---
	Multiprocessor count: 40
	Shared memory per mp: 49152
	Registers per mp: 65536
	Threads in warp: 32
	Max threads per block: 1024
	Max threads dimensions: (1024, 1024, 64)
	Max grid dimensions: 2147483647, 65535, 65535)
*/
#include <iostream>
#include "common.h"

int main(void) {
    cudaDeviceProp prop;
    int count;

    cudaError_t err;

    err = cudaGetDeviceCount(&count);
    HANDLE_ERROR(err);

    for (int i = 0; i < count; i++) {
        err = cudaGetDeviceProperties(&prop, i);
        HANDLE_ERROR(err);

        std::cout<< " --- General information for device " << i << " ---" << std::endl;
        std::cout<< "\tName: " << prop.name << std::endl;
        std::cout<< "\tCompute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout<< "\tClock rate: " << prop.clockRate << std::endl;
        std::cout<< "\tDevice copy overlap: ";
        if (prop.deviceOverlap) {
            std::cout<< "enabled";
        } else {
            std::cout<< "disabled";
        }
        std::cout<< std::endl;
        std::cout<< "\tKernel excition timeout: ";
        if (prop.kernelExecTimeoutEnabled) {
            std::cout<< "enabled";
        } else {
            std::cout<< "disabled";
        }
        std::cout<< std::endl;

        std::cout<< " --- General information for device ---\n";
        std::cout<< "\tTotal globale memory: " << prop.totalGlobalMem << std::endl;
        std::cout<< "\tTotal constant memory: " << prop.totalConstMem << std::endl;
        std::cout<< "\tMax memory pitch: " << prop.memPitch << std::endl;
        std::cout<< "\tTexture alignment: " << prop.textureAlignment << std::endl;

        std::cout<< " --- MP information for device ---\n";
        std::cout<< "\tMultiprocessor count: " << prop.multiProcessorCount << std::endl;
        std::cout<< "\tShared memory per mp: " << prop.sharedMemPerBlock << std::endl;
        std::cout<< "\tRegisters per mp: " << prop.regsPerBlock << std::endl;
        std::cout<< "\tThreads in warp: " << prop.warpSize << std::endl;
        std::cout<< "\tMax threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout<< "\tMax threads dimensions: (" << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
        std::cout<< "\tMax grid dimensions: " << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1]  << ", " << prop.maxGridSize[2] << ")\n";
        std::cout<< std::endl;
    }


    return 0;
}