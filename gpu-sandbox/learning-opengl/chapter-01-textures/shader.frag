#version 460 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;

uniform sampler2D texture0;
uniform sampler2D texture1;

void main() {
    // linearly interpolate between both textures (30% container, 70% awesomeface).
    FragColor = mix(texture(texture0, TexCoord),
                    texture(texture1, TexCoord), 0.7);
}