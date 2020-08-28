#include <toy/present/window.h>
#include <toy/utils/diagnostic.h>
#include <toy/utils/log.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

TOY_NS_OPEN

Window::Window( const std::string& i_title, const gm::Vec2i& i_dimensions )
{
    glfwSetErrorCallback( _ErrorCallback );
    if ( !glfwInit() )
    {
        exit( EXIT_FAILURE );
    }

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 2 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 0 );
    glfwWindowHint( GLFW_VISIBLE, GLFW_TRUE );

    m_handle = glfwCreateWindow( i_dimensions.X(), i_dimensions.Y(), i_title.c_str(), nullptr, nullptr );
    if ( !m_handle )
    {
        glfwTerminate();
        exit( EXIT_FAILURE );
    }

    // Set this class instance as user data available in callbacks.
    glfwSetWindowUserPointer( m_handle, this );

    // Set this window's OpenGL context to be the current one for this thread.
    glfwMakeContextCurrent( m_handle );

    // Number of screen updates to wait from the time glfwSwapBuffers was called before swapping and returning.
    glfwSwapInterval( 1 );
};

void Window::Run()
{
    int width, height;
    glfwGetFramebufferSize( m_handle, &width, &height );
    OnResize( gm::Vec2i( width, height ) );

    glfwSetFramebufferSizeCallback( m_handle, _FrameBufferSizeCallback );
    glfwSetMouseButtonCallback( m_handle, _MouseButtonCallback );
    glfwSetKeyCallback( m_handle, _KeyCallback );
    glfwSetCursorPosCallback( m_handle, _MouseMoveCallback );

    while ( !glfwWindowShouldClose( m_handle ) )
    {
        Render();
        Present();

        glfwSwapBuffers( m_handle );
        glfwPollEvents();
    }
}

/* static */
void Window::_ErrorCallback( int i_error, const char* i_description )
{
    TOY_LOG_ERROR( "Error: %s\n", i_description );
}

/* static */
void Window::_KeyCallback( GLFWwindow* i_glfwWindow, int i_key, int i_scanCode, int i_action, int i_modifiers )
{
    Window* window = static_cast< Window* >( glfwGetWindowUserPointer( i_glfwWindow ) );
    TOY_ASSERT( window );
    if ( i_action == GLFW_PRESS )
    {
        window->OnKeyPress( i_key, i_action, i_modifiers );
    }
}

/* static */
void Window::_MouseMoveCallback( GLFWwindow* i_glfwWindow, double i_xCoord, double i_yCoord )
{
    Window* window = static_cast< Window* >( glfwGetWindowUserPointer( i_glfwWindow ) );
    TOY_ASSERT( window );
    window->OnMouseMove( gm::Vec2i( ( int ) i_xCoord, ( int ) i_yCoord ) );
}

/* static */
void Window::_MouseButtonCallback( GLFWwindow* i_glfwWindow, int i_button, int i_action, int i_modifiers )
{
    Window* window = static_cast< Window* >( glfwGetWindowUserPointer( i_glfwWindow ) );
    TOY_ASSERT( window );
    window->OnMouseButton( i_button, i_action, i_modifiers );
}

/* static */
void Window::_FrameBufferSizeCallback( GLFWwindow* i_glfwWindow, int width, int height )
{
    Window* window = static_cast< Window* >( glfwGetWindowUserPointer( i_glfwWindow ) );
    TOY_ASSERT( window );
    window->OnResize( gm::Vec2i( width, height ) );
}

TOY_NS_CLOSE
