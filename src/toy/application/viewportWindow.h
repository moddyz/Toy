#pragma once

/// \file application/viewportWindow.h

#include <toy/application/window.h>
#include <toy/imaging/lookAtTransform.h>
#include <toy/imaging/perspectiveView.h>
#include <toy/memory/matrix.h>

#include <gm/types/vec2f.h>
#include <gm/types/vec2i.h>
#include <gm/types/vec3f.h>

TOY_NS_OPEN

// Forward declarations.
class CudaGLFrameBuffer;

/// \class ViewportWindow
///
/// A specialized window with a viewport for:
/// - Presenting rendered images.
/// - Offering interactive control over the camera.
class ViewportWindow : public Window
{
public:
    /// Construct a new ViewportWindow with title & window dimensions.
    explicit ViewportWindow( const char* i_title, const gm::Vec2i& i_dimensions );

    //-------------------------------------------------------------------------
    /// \name Rendering
    //-------------------------------------------------------------------------

    /// Derived class should implement logic to writejinto \p o_image.
    virtual void Render( Matrix< gm::Vec3f, Host >& o_image ) = 0;

    //-------------------------------------------------------------------------
    /// \name User Interaction
    //-------------------------------------------------------------------------

    /// Response to a window resize event.
    ///
    /// Internally managed frame buffers will be resized accordingly to match new window size.
    ///
    /// \param i_dimensions The new dimensions (X, Y) of the window.
    virtual void OnResize( const gm::Vec2i& i_dimensions ) override;

    /// Response to a mouse move event.
    ///
    /// \param i_position The new mouse position.
    virtual void OnMouseMove( const gm::Vec2f& i_position ) override;

    /// Response to a scroll event.
    ///
    /// \param i_position The new scroll offset.
    virtual void OnScroll( const gm::Vec2f& i_offset ) override;

    //-------------------------------------------------------------------------
    /// \name Rendering
    //-------------------------------------------------------------------------

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
    //
    virtual void WriteFrame( uint32_t* o_pixels ) override;

    // Camera members.
    LookAtTransform m_cameraTransform;
    PerspectiveView m_cameraView;

    // Cuda <-> GL frame buffer.
    CudaGLFrameBuffer* m_frameBuffer = nullptr;

    // Intermediate frame buffer(s).
    Matrix< gm::Vec3f, Host > m_image;
    Matrix< uint32_t, Host >  m_texture;
};

TOY_NS_CLOSE
