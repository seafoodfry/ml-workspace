#include <glad/glad.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>



class ShaderReadException : public std::exception {
public:
    ShaderReadException(const std::string& message) : m_message(message) {}
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
        throw ShaderReadException(errMsg.str());
    }
    
    return buffer.str();
}

// compileShader expects the output of readShaderCode as input.
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
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        glDeleteShader(shader);
        return -1;
    }

    return shader;
}


Shader::Shader(const char* vertexPath, const char* fragmentPath) {
    // Load the shaders.
    try {
        std::string vertexShaderCode = readShaderCode(vertexPath);
        std::string fragmentShaderCode = readShaderCode(fragmentPath);

    } catch (const std::exception& e) {
        std::cerr << "Shader read exception:\n" << e.what() << std::endl;
        throw;
    }

    // Load and compile the vertex shader.
    GLuint vertexShader = compileShader(vertexShaderCode, GL_VERTEX_SHADER);

    // Load and compile the fragment shader.
    GLuint fragmentShader = compileShader(fragmentShaderCode, GL_FRAGMENT_SHADER);

    // Create and link the shaders to create a shader program.
    std::vector<GLuint> shaders = {vertexShader, fragmentShader};
    GLuint shaderProgram = createShaderProgram(shaders);
    for (const auto& shader: shaders) {
        glDetachShader(shaderProgram, shader);
        glDeleteShader(shader);
    }
}