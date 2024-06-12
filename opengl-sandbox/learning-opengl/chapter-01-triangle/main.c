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

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", nullptr, nullptr);
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


    // Every shader and rendering call after this will use the shader program object.
    glUseProgram(shaderProgam);
    for (const auto& shader: shaders) {
        glDetachShader(shader);
        glDeleteShader(shader);
    }

    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f, 
    };

    unsigned int VBO;
    glGenBuffers(1, &VBO);
    // Bind the VBO buffer to the GL_ARRAY_BUFFER target.
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // Copy the vertex data into the buffer.
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    while (!glfwWindowShouldClose(window)) {
        // Process input.
        processInput(window);

        // Render.
        //glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        //glClearColor(0.96f, 0.51f, 0.19f, 1.0f);

        // Calculate the color based on the elapsed time.
        // initial orange color (0.95f, 0.91f, 0.69f) and the target orange-ish color (0.98f, 0.91f, 0.69f)
        // fmod is the floating point remainder, https://en.cppreference.com/w/cpp/numeric/math/fmod
        // For example, fmod(5.1, 3) = 2.1.
        // The glfwGetTime() function returns the time elapsed since GLFW was initialized, in seconds,
        // as a double value.
        float timeValue = glfwGetTime();
        float transitionRatio = std::fmod(timeValue, TRANSITION_DURATION) / TRANSITION_DURATION;
        float red = 0.95f + (0.98f - 0.95f) * transitionRatio;
        float green = 0.91f - (0.91f - 0.75f) * transitionRatio;
        float blue = 0.69f - (0.69f - 0.18f) * transitionRatio;

        // Render.
        glClearColor(red, green, blue, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteProgram(shaderProgam);
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
    GLUint shader = glCreateShader(shaderType);

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
    GLuint progam = glCreateProgram();

    for (const auto& shader: shaders) {
        glAttachShader(program, shader);
    }
    glLinkProgram(progam);

    // Check if the shader linking process was successful.
    GLuint success;
    glGetProgramiv(progam, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLong[512];
        glGetProgramInfoLog(progam, 512, nullptr, infoLog);
        std::cout<< "ERROR::SHADER::OBJECT::LINKING_FAILED\n" << infoLog << std::endl;
        glDeleteProgram(program);
        return -1;
    }

    return program;
}