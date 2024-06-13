#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <cmath>


const unsigned int WIDTH = 800;
const unsigned int HEIGHT = 600;
const float TRANSITION_DURATION = 3.5f; // Duration of the color transition in seconds


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);


int main() {
    // Instantiate the GLFW window.
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); needed on macos.

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL) {
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