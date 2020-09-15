#pragma once

/// \file application/viewportWindow.h
///
/// A window with a viewport for presenting rendered images.

#include <toy/application/window.h>
#include <toy/memory/matrix.h>
#include <toy/model/lookAtTransform.h>
#include <toy/model/perspectiveView.h>

#include <gm/types/vec2f.h>
#include <gm/types/vec2i.h>
#include <gm/types/vec3f.h>

TOY_NS_OPEN

// Forward declarations.
class CudaGLFrameBuffer;

/// \class ViewportWindow
///
/// A specialized window with a viewport for presenting rendered images.
///
class ViewportWindow : public Window
{
public:
    explicit ViewportWindow( const char* i_title, const gm::Vec2i& i_dimensions );

    /// Resize the internal image buffer.
    virtual void OnResize( const gm::Vec2i& i_dimensions ) override;

    /// Common keyboard shortcuts.
    virtual void OnKeyPress( int i_key, int i_action, int i_modifiers ) override;

    /// Viewport camera movement handlers.
    virtual void OnMouseMove( const gm::Vec2f& i_position ) override;

    /// Viewport camera zoom handler.
    void OnMouseScroll( const gm::Vec2f& i_offset );

    /// Derived class should implement logic to writejinto \p o_image.
    virtual void Render( Matrix< gm::Vec3f, Host >& o_image ) = 0;

    /// Get the current camera view.
    inline const PerspectiveView& GetCameraView() const
    {
        return m_cameraView;
    }

    /// Get the current camera transform.
    inline const LookAtTransform& GetCameraTransform() const
    {
        return m_cameraTransform;
    }

private:
    // Override the default render
    virtual void WriteFrame( uint32_t* o_pixels ) override;

    // Camera members.
    LookAtTransform m_cameraTransform;
    PerspectiveView m_cameraView;

    // CUDA GL Imaging pipeline.
    CudaGLFrameBuffer* m_frameBuffer = nullptr;

    // Frame buffer(s).
    Matrix< gm::Vec3f, Host > m_image;
    Matrix< uint32_t, Host >  m_texture;
};

TOY_NS_CLOSE
