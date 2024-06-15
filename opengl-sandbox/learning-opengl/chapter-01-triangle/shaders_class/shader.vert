#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 ourColor;

uniform float redOffset;
uniform float greenOffset;
uniform float blueOffset;

void main() {
    gl_Position = vec4(aPos, 1.0);
    ourColor = aColor + vec3(redOffset, greenOffset, blueOffset);
}