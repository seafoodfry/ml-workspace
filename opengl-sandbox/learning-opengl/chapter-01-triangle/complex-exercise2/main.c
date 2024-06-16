#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>

#include "shader.h"


const unsigned int WIDTH = 800;
const unsigned int HEIGHT = 600;


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);



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

    // Load, compile the shaders and link them into a shader program.
    Shader shader = Shader("shader.vert", "shader.frag");

    /*
        Configure vertex attributes.
    */
   float vertices[] = {
        // positions         // colors
         0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  // bottom right.
        -0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  // bottom left.
         0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f   // top.
    };

    /*
        1. Bind vertex array objects.
        2. Copy our vertices arrays in a buffer for openGL to use.
        3. Then set the vertex attribute pointers.
    */
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // Position attribute.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Color attribute.
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // The call glVertexAttribPointer() registered VBO as the vertex attribute's bound
    // vertex buffer object, so we can safely unbind now.
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Now we can also unding the VAO so that other VAO calls won't modify this one.
    glBindVertexArray(0);

    // Create a random number engine.
    std::random_device rd; // obtain a random number from hardware.
    std::mt19937 rng(rd()); // seed the generator.
    std::uniform_real_distribution<float> distrPos(0.0f, 0.01f);
    std::uniform_real_distribution<float> distrNeg(-0.01f, 0.0f);
    std::uniform_real_distribution<float> distr(-0.01f, 0.01f);
    //std::uniform_int_distribution<> distr(0, 2); // define the range.

    float stepY = distr(rng);
    float stepX = distr(rng);

    float topPosY = vertices[13];
    float leftPosY = vertices[7];
    float leftPosX = vertices[6];
    float rightPosX = vertices[0];

    while (!glfwWindowShouldClose(window)) {
        // Process input.
        processInput(window);

        // Render.
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw.
        shader.use();

        float timeValue = glfwGetTime();
        float redOffset = static_cast<float>((sin(timeValue * M_PI) / 2.0f) + 0.5f);
        float greenOffset = static_cast<float>((sin(timeValue + 2.0f * M_PI / 3.0f) / 2.0f) + 0.5f);
        float blueOffset = static_cast<float>((sin(timeValue + 4.0f * M_PI / 3.0f) / 2.0f) + 0.5f);
        try {
            shader.setFloat("redOffset", redOffset);
            shader.setFloat("greenOffset", greenOffset);
            shader.setFloat("blueOffset", blueOffset);
        } catch(std::exception& e) {
            std::cerr << "Error setting color offset values: " << e.what() << std::endl;
            return -1;
        }

        // Check if top vertex is going out of bounds, if so generate a new value for stepY
        // that is negative. Also generate a new stepX.
        if (topPosY + stepY >= 1.0) {
            stepY = distrNeg(rng);
            stepX = distr(rng);
        }
        // On the other hand, if the bottom left vertex is going out of bounds from below, then
        // generate a new value for stepY in the positive direction. Also generate a new stepX.
        if (leftPosY + stepY <= -1.0) {
            stepY = distrPos(rng);
            stepX = distr(rng);
        }
        // If the left vertex is going out of bounds from the left, then generate a new positive
        // value for stepX and a new stepY.
        if (leftPosX + stepX <= -1.0) {
            stepX = distrPos(rng);
            stepY = distr(rng);
        }
        // Finally, if the right vertes is going out of bounds from the right, then generate a new
        // stepX is the negative direction and a new stepY.
        if (rightPosX + stepX >= 1.0) {
            stepX = distrNeg(rng);
            stepY = distr(rng);
        }
    
        try {
            float randomZ = 0.0;
            shader.setFloat3f("displacement", stepX, stepY, randomZ);
        } catch (const std::exception& e) {
            std::cerr << "Error setting 'displacement' uniform: " << e.what() << std::endl;
            return -1;
        }


        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    // glDeleteProgram(shader.ID); is now done by the Shader destructor.

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