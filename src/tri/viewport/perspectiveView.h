#pragma once

/// \file viewport/perspectiveView.h

#include <tri/tri.h>

#include <gm/functions/normalize.h>
#include <gm/functions/radians.h>

#include <gm/types/vec3f.h>

TRI_NS_OPEN

/// \class PerspectiveView
///
/// A user-interfacing perspective viewing frustum.
class PerspectiveView
{
public:
    inline explicit PerspectiveView( float i_verticalFov, float i_aspectRatio, float i_focalDistance = 1.0f )
        : m_focalDistance( i_focalDistance )
    {
        // Compute the viewport height from the vertical field of view.
        float verticalFovRadians = gm::Radians( i_verticalFov );

        // This is the important trig ratio which will allow us to compute the viewport height
        // from the focal length.
        float halfViewportHeightOverFocalLength = tan( verticalFovRadians / 2 );
        m_viewportHeight                        = 2.0f * halfViewportHeightOverFocalLength;

        // Compute the width of the virtual viewport.
        m_viewportWidth = i_aspectRatio * m_viewportHeight;
    }

    //-------------------------------------------------------------------------
    /// \name Near plane
    //-------------------------------------------------------------------------

    /// Get the 3D vector matching the near plane width.
    ///
    /// \return Vertical vector of the viewport width.
    inline gm::Vec3f NearHorizontal() const
    {
        return m_focalDistance * m_viewportWidth * gm::Vec3f( 1, 0, 0 );
    }

    /// Get the 3D vector matching the near plane height.
    ///
    /// \return Vertical vector of the viewport height.
    inline gm::Vec3f NearVertical() const
    {
        return m_focalDistance * m_viewportHeight * gm::Vec3f( 0, 1, 0 );
    }

    /// Get the 3D coordinate of the bottom left corner of the near plane.
    ///
    /// \return The bottom left coordinate of the viewport plane.
    inline gm::Vec3f NearBottomLeft() const
    {
        return m_focalDistance * gm::Vec3f( 0, 0, 1 ) - ( NearHorizontal() * 0.5 ) - ( NearVertical() * 0.5 );
    }

private:
    // The distance between the camera origin and the focal plane.
    float m_focalDistance = 1.0f;

    // The fixed height of the virtual viewport.
    float m_viewportHeight = 2.0f;

    // Variable width of the virtual viewport.
    float m_viewportWidth = 0.0f;
};

TRI_NS_CLOSE
