/*
When you launch a kernel function from the host (CPU) code, the arguments passed to the kernel
are copied from the host memory to the device (GPU) memory.
Since references are not actual objects but rather aliases to existing objects,
they cannot be copied directly to the device memory.
*/
#include <iostream>
#include "common.h"

__global__ void add(int a, int b, int* c) {
    // Store the result of a+b in the memory pointed to by c.
    *c = a + b;
}

int main(void) {
    int c;
    int* dev_c;

    /* &dev_c takes the address of the dev_c pointer itself. So, &dev_c is of type int**,
    which is a pointer to a pointer to an integer.
    The (void**) cast is necessary because cudaMalloc expects a pointer to a void* pointer
    as its first argument.
    */
   cudaError_t err = cudaMalloc((void**)&dev_c, sizeof(int));
    HANDLE_ERROR(err);

    add<<<1,1>>>(2, 7, dev_c);

    err = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    HANDLE_ERROR(err);

    printf("2 + 7 = %d\n", c);
    cudaFree(dev_c);

    return 0;
}