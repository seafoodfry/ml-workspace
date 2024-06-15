#pragma once

#include <glad/glad.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


class Shader {
public:
    explicit Shader(const char* vertexPath, const char* fragmentPath);
    ~Shader();

    // Use to activate the shader.
    void use();

    // Utility functions for uniform values.
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setFloat4f(const std::string& name, float val1, float val2, float val3, float val4) const;

private:
    // Program ID.
    GLuint ID;

    GLint getUniformLocation(const std::string& name) const;
};