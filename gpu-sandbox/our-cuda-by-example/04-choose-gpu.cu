#include <iostream>
#include "common.h"

int main(void) {
    cudaDeviceProp prop;
    int dev;
    cudaError_t err;

    err = cudaGetDevice(&dev);
    HANDLE_ERROR(err);

    std::cout<< "ID of current device: " << dev << std::endl;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 7;
    prop.minor = 0;
    err = cudaChooseDevice(&dev, &prop);
    HANDLE_ERROR(err);

    std::cout<< "ID of device closest to compute capability 7.0: " << dev << std::endl;

    err = cudaSetDevice(dev);
    HANDLE_ERROR(err);

    return  0;
}