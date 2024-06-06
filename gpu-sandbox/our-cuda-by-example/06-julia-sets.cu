#include <iostream>
#include "common.h"

int main(void) {
    CPUBitMap bitmap(DIM, DIM);
    unsigned char* ptr = bitmap.get_ptr();

    kernel(ptr);

    bitmap.display_and_exit();

    return 0;
}