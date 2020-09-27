#pragma once

/// \file application/window.h
///
/// An application window for presentation purposes.

#include <toy/application/mouse.h>
#include <toy/memory/matrix.h>

#include <gm/types/vec2f.h>
#include <gm/types/vec2i.h>

// Forward declarations.
class GLFWwindow;

TOY_NS_OPEN

// Forward declarations.
class CudaGLFrameBuffer;

/// \class Window
///
/// Base class for initializing a window for presenting rendered frames.
class Window
{
public:
    /// Initialize the application window.
    ///
    /// \param i_title The title of the application window.
    /// \param i_size The title of the application window.
    explicit Window( const char* i_title, const gm::Vec2i& i_size );
    virtual ~Window();

    /// Execute the main render loop.
    void Run();

    /// Get the 2D size of the current window.
    ///
    /// \return 2D size.
    gm::Vec2i GetSize() const;

protected:
    /// Executes the render computation, and write into \p o_frameData.
    ///
    /// \p o_frameData can be treated like a CUDA device buffer.
    ///
    /// The size of \p o_frameBuffer is Width * Height * sizeof( uint32_t ).
    /// It is a RGBA buffer with 8 bits or 1 byte allocated for each channel.
    ///
    /// \param o_frameData The frame buffer.
    virtual void WriteFrame( uint32_t* o_frameData ) = 0;

    /// Respond to a window resize event.
    ///
    /// \param i_dimensions The new dimensions for this window.
    inline virtual void OnResize( const gm::Vec2i& i_dimensions )
    {
        /* no-op */
    }

    /// Respond to a key press event.
    ///
    /// \param i_key The ID of the key which was pressed.
    /// \param i_action ???
    /// \param i_modifiers If any keyboard modifiers are active (ctrl, alt, shift).
    inline virtual void OnKeyPress( int i_key, int i_action, int i_modifiers )
    {
        /* no-op */
    }

    /// Respond to a mouse move event.
    ///
    /// \param i_position New mouse position.
    inline virtual void OnMouseMove( const gm::Vec2f& i_position )
    {
        /* no-op */
    }

    /// Respond to a mouse button event.
    ///
    /// \param i_key The ID of the mouse button which was pressed.
    /// \param i_action ???
    /// \param i_modifiers If any keyboard modifiers are active (ctrl, alt, shift).
    inline virtual void OnMouseButton( int i_button, int i_action, int i_modifiers )
    {
        /* no-op */
    }

    /// Respond to a scroll event.
    ///
    /// \param i_offset The scroll offset in pixel units.
    inline virtual void OnScroll( const gm::Vec2f& i_offset )
    {
        /* no-op */
    }

    // ------------------------------------------------------------------------
    /// \name User input
    // ------------------------------------------------------------------------

    inline MouseButton GetMouseButtonPressed() const
    {
        return m_mouseButtonPressed;
    }

    inline const gm::Vec2f& GetLastMousePosition() const
    {
        return m_lastMousePosition;
    }

private:
    // Resize the frame buffer, and call client OnResize.
    void _Resize( const gm::Vec2i& i_dimensions );

    // Query the current mouse position.
    gm::Vec2f _GetMousePosition() const;

    // GLFW callbacks.
    static void _ErrorCallback( int i_error, const char* i_description );
    static void _KeyCallback( GLFWwindow* i_glfwWindow, int i_key, int i_scanCode, int i_action, int i_modifiers );
    static void _MouseMoveCallback( GLFWwindow* i_glfwWindow, double i_xCoord, double i_yCoord );
    static void _MouseButtonCallback( GLFWwindow* i_glfwWindow, int i_button, int i_action, int i_modifiers );
    static void _ScrollCallback( GLFWwindow* i_glfwWindow, double i_xOffset, double i_yOffset );
    static void _FrameBufferSizeCallback( GLFWwindow* i_glfwWindow, int i_width, int i_height );

    // Handle to the underlying GLFW window instance.
    GLFWwindow* m_handle = nullptr;

    // CUDA GL Imaging pipeline.
    CudaGLFrameBuffer* m_frameBuffer = nullptr;

    // User input states.
    gm::Vec2f   m_lastMousePosition;
    MouseButton m_mouseButtonPressed;
};

TOY_NS_CLOSE
