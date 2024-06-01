/*
    g++ -c opengl_test.cpp -I/usr/include/GL

    g++ -o opengl_test opengl_test.o -lGL -lGLU -lglut
*/
#include <GL/glut.h>

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    
    glBegin(GL_TRIANGLES);
    glColor3f(1.0f, 0.0f, 0.0f);  // Red
    glVertex2f(-0.5f, -0.5f);
    glColor3f(0.0f, 1.0f, 0.0f);  // Green
    glVertex2f(0.5f, -0.5f);
    glColor3f(0.0f, 0.0f, 1.0f);  // Blue
    glVertex2f(0.0f, 0.5f);
    glEnd();
    
    glFlush();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutCreateWindow("OpenGL Test");
    glutDisplayFunc(display);
    
    glutMainLoop();
    
    return 0;
}