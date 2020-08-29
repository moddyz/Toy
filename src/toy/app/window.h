#pragma once

/// \file app/window.h
///
/// An application window for presentation purposes.

#include <toy/memory/matrix.h>

#include <gm/types/vec2i.h>

// Forward declarations.
class GLFWwindow;

TOY_NS_OPEN

/// \class Window
///
/// Base class for initializing a window for presenting rendered frames.
class Window
{
public:
    /// Initialize the application window.
    ///
    /// \param i_title The title of the application window.
    explicit Window( const char* i_title, const gm::Vec2i& i_dimensions );
    virtual ~Window();

    /// Execute the main render loop.
    void Run();

protected:
    /// Executes the render process to produce a single frame.
    virtual void Render() = 0;

    /// Gets the image from the last render.
    ///
    /// \param o_image Image buffer to fill.
    virtual void GetImage( Matrix< uint32_t, Host >& o_image ) = 0;

    /// Respond to a window resize event.
    ///
    /// \param i_dimensions The new dimensions for this window.
    virtual void OnResize( const gm::Vec2i& i_dimensions );

    /// Respond to a key press event.
    ///
    /// \param i_key The ID of the key which was pressed.
    /// \param i_action ???
    /// \param i_modifiers If any keyboard modifiers are active (ctrl, alt, shift).
    virtual void OnKeyPress( int i_key, int i_action, int i_modifiers ){};

    /// Respond to a mouse move event.
    ///
    /// \param i_position New mouse position.
    virtual void OnMouseMove( const gm::Vec2i& i_position ){};

    /// Respond to a mouse button event.
    ///
    /// \param i_key The ID of the mouse button which was pressed.
    /// \param i_action ???
    /// \param i_modifiers If any keyboard modifiers are active (ctrl, alt, shift).
    virtual void OnMouseButton( int i_button, int i_action, int i_modifiers ){};

private:
    // Present the last rendered frame in the window.
    void _Present();

    // GLFW callbacks.
    static void _ErrorCallback( int i_error, const char* i_description );
    static void _KeyCallback( GLFWwindow* i_glfwWindow, int i_key, int i_scanCode, int i_action, int i_modifiers );
    static void _MouseMoveCallback( GLFWwindow* i_glfwWindow, double i_xCoord, double i_yCoord );
    static void _MouseButtonCallback( GLFWwindow* i_glfwWindow, int i_button, int i_action, int i_modifiers );
    static void _FrameBufferSizeCallback( GLFWwindow* i_glfwWindow, int i_width, int i_height );

    // Handle to the underlying GLFW window instance.
    GLFWwindow* m_handle = nullptr;

    gm::Vec2i                m_frameBufferSize;
    uint32_t                 m_frameBufferTexture = 0;
    Matrix< uint32_t, Host > m_image;
};

TOY_NS_CLOSE

