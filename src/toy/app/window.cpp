#include <toy/app/window.h>
#include <toy/utils/diagnostic.h>
#include <toy/utils/log.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <GL/gl.h>

TOY_NS_OPEN

Window::Window( const char* i_title, const gm::Vec2i& i_dimensions )
{
    glfwSetErrorCallback( _ErrorCallback );
    if ( !glfwInit() )
    {
        exit( EXIT_FAILURE );
    }

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 2 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 0 );
    glfwWindowHint( GLFW_VISIBLE, GLFW_TRUE );

    m_handle = glfwCreateWindow( i_dimensions.X(), i_dimensions.Y(), i_title, nullptr, nullptr );
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
}

Window::~Window()
{
    glfwDestroyWindow( m_handle );
    glfwTerminate();
}

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
        _Present();

        glfwSwapBuffers( m_handle );
        glfwPollEvents();
    }
}

void Window::OnResize( const gm::Vec2i& i_dimensions )
{
    m_image.Resize( i_dimensions.Y(), i_dimensions.X() );
    m_frameBufferSize = i_dimensions;
}

void Window::_Present()
{
    GetImage( m_image );
    if ( m_frameBufferTexture == 0 )
    {
        glGenTextures( 1, &m_frameBufferTexture );
    }

    glBindTexture( GL_TEXTURE_2D, m_frameBufferTexture );
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_UNSIGNED_BYTE;
    glTexImage2D( GL_TEXTURE_2D,
                  0,
                  texFormat,
                  m_frameBufferSize.X(),
                  m_frameBufferSize.Y(),
                  0,
                  GL_RGBA,
                  texelType,
                  m_image.GetBuffer() );

    glDisable( GL_LIGHTING );
    glColor3f( 1, 1, 1 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glEnable( GL_TEXTURE_2D );
    glBindTexture( GL_TEXTURE_2D, m_frameBufferTexture );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    glDisable( GL_DEPTH_TEST );

    glViewport( 0, 0, m_frameBufferSize.X(), m_frameBufferSize.Y() );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0.f, ( float ) m_frameBufferSize.X(), 0.f, ( float ) m_frameBufferSize.Y(), -1.f, 1.f );

    glBegin( GL_QUADS );
    {
        glTexCoord2f( 0.f, 0.f );
        glVertex3f( 0.f, 0.f, 0.f );

        glTexCoord2f( 0.f, 1.f );
        glVertex3f( 0.f, ( float ) m_frameBufferSize.Y(), 0.f );

        glTexCoord2f( 1.f, 1.f );
        glVertex3f( ( float ) m_frameBufferSize.X(), ( float ) m_frameBufferSize.Y(), 0.f );

        glTexCoord2f( 1.f, 0.f );
        glVertex3f( ( float ) m_frameBufferSize.X(), 0.f, 0.f );
    }
    glEnd();
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
