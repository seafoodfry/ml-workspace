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

    // Static method used for glut callbacks.
    static CPUBitmap** get_bitmap_ptr(void) {
        // When a variable is declared as static inside a function, it retains its value between function
        // calls. This means that the variable is initialized only once (the first time the function is
        // called) and retains its value across subsequent calls to the function.
        static CPUBitmap* gBitmap;
        return &gBitmap;
    }

    // void(*e)(void*) is a parameter declaring a function pointer named e.
    // The function pointer e has the same signature as the function pointer
    // bitmapExit.
    void display_and_exit( void(*e)(void*) = NULL ) {
        CPUBitmap** bitmap = get_bitmap_ptr();
        // Retrieve gBitmap and store the current instance on it.
        *bitmap = this;
        bitmapExit = e;

        int argc = 1;
        char *argv[1] = {(char*)"Something"};
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
        glutInitWindowSize(x, y);
        glutCreateWindow("bitmap");
        glutKeyboardFunc(Key);
        glutDisplayFunc(Draw);
        glutMainLoop();
    }

    /* Static methods used for glut callbacks. */
    static void Key(unsigned char key, int x, int y) {
        switch (key) {
            case 27:
                CPUBitmap* bitmap = *(get_bitmap_ptr());
                if (bitmap->dataBlock != NULL && bitmap->bitmapExit != NULL) {
                    bitmap->bitmapExit(bitmap->dataBlock);
                }
                exit(0);
        }
    }

    static void Draw(void) {
        CPUBitmap* bitmap = *(get_bitmap_ptr());
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
        glFlush();
    }
};
