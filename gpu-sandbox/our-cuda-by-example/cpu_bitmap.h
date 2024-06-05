#pragma once

/* On Linux, include the system's copy of glut.h, glext.h, and glx.h */
#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/glx.h>

#define GET_PROC_ADDRESS( str ) glXGetProcAddress( (const GLubyte *)str )

struct  CPUBitmap {
    unsigned char* pixels;
    int x, y;
    void* dataBlock;
    // Function pointer.
    void (*bitmapExit)(void*);

    CPUBitmap(int width, int height, void* d = NULL) {
        pixels = new unsigned char[width * height * 4];
        x = width;
        y = height;
        dataBlock = d;
    }
    
    ~CPUBitmap() {
        delete []pixels;
    }

    unsigned char* get_ptr(void) const { return pixels; }
    long image_size(void) const { return x*y*4; }

    // void(*e)(void*) is a parameter declaring a function pointer named e.
    // The function pointer e has the same signature as the function pointer
    // bitmapExit.
    void display_and_exit( void(*e)(void*) = NULL ) {

    }
};
