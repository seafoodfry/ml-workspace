/*
    Here we build upon the basic triangle program and now draw 2 triangles
    next to each other using glDrawArrays() by adding more vertices to our data.
*/
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>


const unsigned int WIDTH = 800;
const unsigned int HEIGHT = 600;
const float TRANSITION_DURATION = 3.5f; // Duration of the color transition in seconds


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
std::string readShaderCode(const std::string& filePath);
GLuint compileShader(const std::string& shaderCode, GLenum shaderType);
GLuint createShaderProgram(const std::vector<GLuint>& shaders);


int main() {
    // Instantiate the GLFW window.
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); needed on macos.

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL: 2 triangles", nullptr, nullptr);
    if (window == nullptr) {
        std::cout<< "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    // Register the callback function for resizing.
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Initialize GLAD to manage function pointer for openGL.
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout<< "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Load and compile the vertex shader.
    std::string vertexShaderCode = readShaderCode("shader.vert");
    GLuint vertexShader = compileShader(vertexShaderCode, GL_VERTEX_SHADER);

    // Load and compile the fragment shader.
    std::string fragmentShaderCode = readShaderCode("shader.frag");
    GLuint fragmentShader = compileShader(fragmentShaderCode, GL_FRAGMENT_SHADER);

    // Create and link the shaders to create a shader program.
    std::vector<GLuint> shaders = {vertexShader, fragmentShader};
    GLuint shaderProgram = createShaderProgram(shaders);
    for (const auto& shader: shaders) {
        glDetachShader(shaderProgram, shader);
        glDeleteShader(shader);
    }

    /*
        Configure vertex attributes.
    */
    float vertices[] = {
        // first triangle
        -0.9f, -0.5f, 0.0f,  // left 
        -0.0f, -0.5f, 0.0f,  // right
        -0.45f, 0.5f, 0.0f,  // top 
        // second triangle
        0.0f, -0.5f, 0.0f,  // left
        0.9f, -0.5f, 0.0f,  // right
        0.45f, 0.5f, 0.0f   // top 
    };

    /*
        1. Bind vertex array objects.
        2. Copy our vertices arrays in a buffer for openGL to use.
        3. Then set the vertex attribute pointers.
    */
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);

    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    // Bind the VBO buffer to the GL_ARRAY_BUFFER target.
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // Copy the vertex data into the buffer.
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // The second argument, the `size` stays the same since that's the number of components per index.
    // The stride stays the same ->3 float / vertex.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // The call glVertexAttribPointer() registered VBO as the vertex attribute's bound
    // vertex buffer object, so we can safely unbind now.
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Now we can also unding the VAO so that other VAO calls won't modify this one.
    glBindVertexArray(0);


    while (!glfwWindowShouldClose(window)) {
        // Process input.
        processInput(window);

        // Render.
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw.
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    (void)window; // Prevent the -Werror=unused-parameter error during compilation.
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

// Function to read shader code from file.
std::string readShaderCode(const std::string& filePath) {
    std::ifstream file(filePath);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

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
        std::cout<< "ERROR::SHADER::OBJECT::LINKING_FAILED\n" << infoLog << std::endl;
        glDeleteProgram(program);
        return -1;
    }

    return program;
}