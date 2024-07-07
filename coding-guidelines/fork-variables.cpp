/*
    g++ -o fork-variables.out fork-variables.cpp
*/
#include <cstdio>
#include <unistd.h>
//#include <sys/types.h> unistd.h also provides pid_t.

int main(int argc, char* argv[]) {
    int x = 100;

    pid_t pid = fork();
    if (pid < 0) {  // For failed.
        fprintf(stderr, "fork failed\n");
        return -1;
    } else if (pid == 0) {  // The child process.
        printf("[child] initial value of x: %d\n", x);
        x = 200;
        printf("[child] new value of x: %d\n", x);
    } else {  // pid > 0 and equtl to the child's PID.
        printf("[parent] initial value of x: %d\n", x);
        x = 300;
        printf("[parent] new value of x: %d\n", x);
    }
    return 0;
}