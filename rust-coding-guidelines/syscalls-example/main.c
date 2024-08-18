#define _GNU_SOURCE  // Becuase of the SYS_write and other syscall bits.
#include <linux/kernel.h>
#include <sys/syscall.h>
//#include <unistd.h>  // This gets rid of the implicit definition warning.
#include <string.h>
#include <stdio.h>

int main() {
    const char* msg = "Hello from a syscall in C!\n";
    long result = syscall(SYS_write, 1, msg, strlen(msg));
    printf("length of C message: %ld\n", strlen(msg));
    printf("Syscall result: %ld\n", result);
    return 0;
}