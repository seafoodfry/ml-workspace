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
    ~Shader() = default;

    // Program ID.
    unsigned int ID;

    // Use to activate the shader.
    void use();

    // Utility functions for uniform values.
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;

private:
};