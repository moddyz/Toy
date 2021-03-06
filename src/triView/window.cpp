#include <tri/base/diagnostic.h>
#include <tri/base/log.h>

#include <tri/viewport/cudaGLFrameBuffer.h>

#include <tri/application/window.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// GLFW >3.2 defines this.
#ifndef GLFW_TRUE
#    define GLFW_TRUE 1
#endif

#include <GL/glew.h>

Window::Window(const char* i_title, const gm::Vec2i& i_size)
{
    glfwSetErrorCallback(_ErrorCallback);
    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    m_handle =
        glfwCreateWindow(i_size.X(), i_size.Y(), i_title, nullptr, nullptr);
    if (!m_handle) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Set this class instance as user data available in callbacks.
    glfwSetWindowUserPointer(m_handle, this);

    // Set this window's OpenGL context to be the current one for this thread.
    glfwMakeContextCurrent(m_handle);

    // Number of screen updates to wait from the time glfwSwapBuffers was called
    // before swapping and returning.
    glfwSwapInterval(1);

    // Initialize GLEW.
    if (glewInit() != GLEW_OK) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
}

Window::~Window()
{
    glfwDestroyWindow(m_handle);
    glfwTerminate();
}

void
Window::Run()
{
    _Resize(GetSize());

    glfwSetFramebufferSizeCallback(m_handle, _FrameBufferSizeCallback);
    glfwSetMouseButtonCallback(m_handle, _MouseButtonCallback);
    glfwSetKeyCallback(m_handle, _KeyCallback);
    glfwSetCursorPosCallback(m_handle, _MouseMoveCallback);
    glfwSetScrollCallback(m_handle, _ScrollCallback);

    while (!glfwWindowShouldClose(m_handle)) {
        TRI_ASSERT(m_frameBuffer != nullptr);

        // CUDA Computation step.
        uint32_t* frameData = m_frameBuffer->WriteFrameBegin();
        WriteFrame(frameData);
        m_frameBuffer->WriteFrameEnd();

        // Display the computed frame.
        m_frameBuffer->DrawFrame();

        glfwSwapBuffers(m_handle);
        glfwPollEvents();
    }
}

gm::Vec2i
Window::GetSize() const
{
    int width, height;
    glfwGetFramebufferSize(m_handle, &width, &height);
    return gm::Vec2i(width, height);
}

void
Window::_Resize(const gm::Vec2i& i_size)
{
    TRI_ASSERT(i_size.X() != 0 & i_size.Y() != 0);

    if (m_frameBuffer == nullptr) {
        m_frameBuffer = new CUDAGLFrameBuffer(i_size.X(), i_size.Y());
    } else if (m_frameBuffer->GetWidth() != i_size.X() ||
               m_frameBuffer->GetHeight() != i_size.Y()) {
        delete m_frameBuffer;
        m_frameBuffer = new CUDAGLFrameBuffer(i_size.X(), i_size.Y());
    }

    // Then call derived.
    OnResize(i_size);
}

gm::Vec2f
Window::_GetMousePosition() const
{
    double x, y;
    glfwGetCursorPos(m_handle, &x, &y);
    return gm::Vec2f(x, y);
}

/* static */
void
Window::_ErrorCallback(int i_error, const char* i_description)
{
    TRI_LOG_ERROR("Error: %s\n", i_description);
}

/* static */
void
Window::_KeyCallback(GLFWwindow* i_glfwWindow,
                     int i_key,
                     int i_scanCode,
                     int i_action,
                     int i_modifiers)
{
    Window* window =
        static_cast<Window*>(glfwGetWindowUserPointer(i_glfwWindow));
    TRI_ASSERT(window);
    if (i_action == GLFW_PRESS || i_action == GLFW_REPEAT) {
        window->OnKeyPress(i_key, i_action, i_modifiers);
    }
}

/* static */
void
Window::_MouseMoveCallback(GLFWwindow* i_glfwWindow,
                           double i_xCoord,
                           double i_yCoord)
{
    Window* window =
        static_cast<Window*>(glfwGetWindowUserPointer(i_glfwWindow));
    TRI_ASSERT(window);
    gm::Vec2f mousePosition(i_xCoord, i_yCoord);
    window->OnMouseMove(mousePosition);
    window->m_lastMousePosition = mousePosition;
}

/* static */
void
Window::_MouseButtonCallback(GLFWwindow* i_glfwWindow,
                             int i_button,
                             int i_action,
                             int i_modifiers)
{
    Window* window =
        static_cast<Window*>(glfwGetWindowUserPointer(i_glfwWindow));
    TRI_ASSERT(window);

    // Update mouse mouse pressed state.
    MouseButton mouseButton = MouseButton_None;
    switch (i_button) {
        case GLFW_MOUSE_BUTTON_LEFT:
            mouseButton = MouseButton_Left;
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            mouseButton = MouseButton_Middle;
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            mouseButton = MouseButton_Right;
            break;
    }

    if (i_action == GLFW_PRESS) {
        window->m_mouseButtonPressed |= mouseButton;
    } else {
        window->m_mouseButtonPressed &= ~mouseButton;
    }

    // Call derived function.
    window->OnMouseButton(i_button, i_action, i_modifiers);

    // Update last mouse position state.
    window->m_lastMousePosition = window->_GetMousePosition();
}

/* static */
void
Window::_ScrollCallback(GLFWwindow* i_glfwWindow,
                        double i_xOffset,
                        double i_yOffset)
{
    Window* window =
        static_cast<Window*>(glfwGetWindowUserPointer(i_glfwWindow));
    TRI_ASSERT(window);

    // Call derived function.
    gm::Vec2f mouseScroll(i_xOffset, i_yOffset);
    window->OnScroll(mouseScroll);
}

/* static */
void
Window::_FrameBufferSizeCallback(GLFWwindow* i_glfwWindow,
                                 int width,
                                 int height)
{
    Window* window =
        static_cast<Window*>(glfwGetWindowUserPointer(i_glfwWindow));
    TRI_ASSERT(window);
    window->_Resize(gm::Vec2i(width, height));
}
