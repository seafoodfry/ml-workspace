#include <iostream>
#include "common.h"

#define N 10000

__global__ void add(int* a, int* b, int* c) {
    int tid = blockIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    cudaError_t err;

    // Allocate memory on the GPU.
    err = cudaMalloc((void**)&dev_a, N*sizeof(int));
    HANDLE_ERROR(err);
    err = cudaMalloc((void**)&dev_b, N*sizeof(int));
    HANDLE_ERROR(err);
    err = cudaMalloc((void**)&dev_c, N*sizeof(int));

    // Fill in arrays.
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    // Copy arrays into the GPU.
    err = cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    HANDLE_ERROR(err);
    err = cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    HANDLE_ERROR(err);

    add<<<N,1>>>(dev_a, dev_b, dev_c);

    // Copy array c from GPU to host.
    err = cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    HANDLE_ERROR(err);

    // Display results.
    for (int i=0; i<N; i++) {
        if (i < 10) {
            std::cout<< a[i] << " + " << b[i] << " = " << c[i] << std::endl; 
        }
    }

    // Free allocated memory.
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}