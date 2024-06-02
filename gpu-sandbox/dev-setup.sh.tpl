#!/bin/bash
set -x

sudo yum update -y
sudo yum install -y vim
# Install compilers, make, etc.
sudo yum groupinstall -y "Development Tools"
# FreeGLUT library (which includes GLUT) and the Mesa OpenGL development packages
# (which include the necessary header files and libraries for OpenGL and GLX).
# Necessary header files will be in /usr/include/GL
sudo yum install -y freeglut-devel mesa-libGL-devel

sudo yum install -y xorg-x11-xauth xorg-x11-apps glx-utils