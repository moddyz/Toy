#pragma once

/// \file present/window.h
///
/// An application window for presentation purposes.

#include <toy/toy.h>

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
    Window( const std::string& i_title, const gm::Vec2i& i_dimensions );
    ~Window();

    /// Execute the main render loop.
    void Run();

    /// Executes the render process to produce a single frame.
    virtual void Render() = 0;

    /// Present the last rendered frame in the window.
    virtual void Present() = 0;

protected:
    /// Respond to a window resize event.
    ///
    /// \param i_dimensions The new dimensions for this window.
    virtual void OnResize( const gm::Vec2i& i_dimensions ){};

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
    // GLFW callbacks.
    static void _ErrorCallback( int i_error, const char* i_description );
    static void _KeyCallback( GLFWwindow* i_glfwWindow, int i_key, int i_scanCode, int i_action, int i_modifiers );
    static void _MouseMoveCallback( GLFWwindow* i_glfwWindow, double i_xCoord, double i_yCoord );
    static void _MouseButtonCallback( GLFWwindow* i_glfwWindow, int i_button, int i_action, int i_modifiers );
    static void _FrameBufferSizeCallback( GLFWwindow* i_glfwWindow, int i_width, int i_height );

    GLFWwindow* m_handle = nullptr;
};

TOY_NS_CLOSE

