#include <glad/glad.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "shader.h"


class ShaderException : public std::exception {
public:
    ShaderException(const std::string& message) : m_message(message) {}
    const char* what() const noexcept override { return m_message.c_str(); }

private:
    std::string m_message;
};

// Function to read shader code from file.
std::string readShaderCode(const std::string& filePath) {
    // By default, std::ifstream objects do not throw exceptions when encountering errors.
    // Instead, they set internal error state flags (failbit, badbit, or eofbit) that can be checked using
    // functions like fail(), bad(), or eof().
    // This makes it so that the program will actually throw exceptions if these happen.
    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    
    std::stringstream buffer;
    try {
        file.open(filePath);
        buffer << file.rdbuf();
        file.close();
    } catch (std::ifstream::failure& e) {
        std::stringstream errMsg;
        if (!file.is_open()) {
            errMsg << "Failed to open shader file: " << filePath << "\n";
        } else {
            errMsg << "Failed reading shader file: " << filePath << "\n";
        }
        errMsg << "Error Details: " << e.what();
        throw ShaderException(errMsg.str());
    }
    
    return buffer.str();
}

// compileShader expects the output of readShaderCode() as input.
GLuint compileShader(const std::string& shaderCode, GLenum shaderType) {
    GLuint shader = glCreateShader(shaderType);

    const char* shaderSource = shaderCode.c_str();
    glShaderSource(shader, 1, &shaderSource, nullptr);
    glCompileShader(shader);

    // Check if compilation of the vertex shader was successful.
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);

        std::stringstream errorMsg;
        errorMsg << "Shader compilation failed:\n";
        errorMsg << "Shader type: ";
        switch (shaderType) {
            case GL_VERTEX_SHADER:
                errorMsg << "Vertex Shader\n";
                break;
            case GL_FRAGMENT_SHADER:
                errorMsg << "Fragment Shader\n";
                break;
            // Add more cases for other shader types if needed.
            default:
                errorMsg << "Unknown Shader Type\n";
                break;
        }
        errorMsg << "Error details:\n" << infoLog;

        glDeleteShader(shader);
        throw ShaderException(errorMsg.str());
    }

    return shader;
}

// createShaderProgram takes as input a vector of outputs from compileShader().
GLuint createShaderProgram(const std::vector<GLuint>& shaders) {
    GLuint program = glCreateProgram();

    for (const auto& shader: shaders) {
        glAttachShader(program, shader);
    }
    glLinkProgram(program);

    // Check if the shader linking process was successful.
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);

        std::stringstream errMsg;
        errMsg << "Shader linking failed:\n";
        errMsg << "Error details:\n" << infoLog;

        glDeleteProgram(program);
        throw ShaderException(errMsg.str());
    }

    return program;
}


Shader::Shader(const char* vertexPath, const char* fragmentPath) {
    // Load the shaders.
    std::string vertexShaderCode, fragmentShaderCode;
    try {
        vertexShaderCode = readShaderCode(vertexPath);
        fragmentShaderCode = readShaderCode(fragmentPath);

    } catch (const std::exception& e) {
        std::cerr << "Shader read exception:\n" << e.what() << std::endl;
        throw;
    }

    // Load and compile the shaders.
    GLuint vertexShader, fragmentShader;
    try {
        vertexShader = compileShader(vertexShaderCode, GL_VERTEX_SHADER);
        fragmentShader = compileShader(fragmentShaderCode, GL_FRAGMENT_SHADER);

    } catch(std::exception& e) {
        std::cerr << "Shader compilation exception:\n" << e.what() << std::endl;
        throw;
    }

    // Create and link the shaders to create a shader program.
    std::vector<GLuint> shaders = {vertexShader, fragmentShader};
    try {
        ID = createShaderProgram(shaders);
    } catch(std::exception& e) {
        std::cerr << "Shader linking exception:\n" << e.what() << std::endl;
        throw;
    }
    for (const auto& shader: shaders) {
        glDetachShader(ID, shader);
        glDeleteShader(shader);
    }
}

Shader::~Shader() {
    glDeleteProgram(ID);
}

void Shader::use() {
    glUseProgram(ID);
}

GLint Shader::getUniformLocation(const std::string& name) const {
    GLint location = glGetUniformLocation(ID, name.c_str());
    if (location == -1) {
        throw ShaderException("Invalid uniform location: " + name);
    }
    return location;
}

void Shader::setBool(const std::string& name, bool value) const {
    GLint location = getUniformLocation(name);
    glUniform1i(location, static_cast<GLint>(value));
}

void Shader::setInt(const std::string& name, int value) const {
    GLint location = getUniformLocation(name);
    glUniform1i(location, static_cast<GLint>(value));
}

void Shader::setFloat(const std::string& name, float value) const {
    GLint location = getUniformLocation(name);
    glUniform1f(location, value);
}

void Shader::setFloat4f(const std::string& name, float x, float y, float z, float w) const {
    GLint location = getUniformLocation(name);
    glUniform4f(location, x, y, z, w);
}

void Shader::setFloat3f(const std::string& name, float x, float y, float z) const {
    GLint location = getUniformLocation(name);
    glUniform3f(location, x, y, z);
}