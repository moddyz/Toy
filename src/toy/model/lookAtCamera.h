#pragma once

/// \file model/camera.h
///
/// A camera model with a origin (position), look at position, and

#include <toy/model/camera.h>

#include <gm/functions/crossProduct.h>
#include <gm/functions/lookAt.h>
#include <gm/functions/normalize.h>
#include <gm/functions/radians.h>

#include <gm/types/vec3f.h>

TOY_NS_OPEN

/// \class LookAtCamera
///
/// A camera with a origin, look at, and up vector.
class LookAtCamera : public Camera
{
public:
    /// Construct a camera with transformation, projection, and focal parameters.
    ///
    /// \param i_origin The position of the camera origin, or eye.
    /// \param i_lookAt The point of camera eye is focused onto.
    /// \param i_up The up-vector of the camera.
    /// \param i_verticalFov The vertical field of view, in degrees.  This is the angle formed
    /// from the camera origin towards the viewport height.
    /// \param i_aspectRatio Ratio of the width against the height of the rendered image.
    /// \param i_focalDistance The distance between the camera origin and the focal plane where
    /// objects are in perfect focus.
    inline explicit LookAtCamera( const gm::Vec3f& i_origin,
                                  const gm::Vec3f& i_lookAt,
                                  const gm::Vec3f& i_up,
                                  float            i_verticalFov,
                                  float            i_aspectRatio,
                                  float            i_focalDistance = 1.0f )
        : m_origin( i_origin )
        , m_lookAt( i_lookAt )
        , m_up( i_up )
        , m_aspectRatio( i_aspectRatio )
        , m_focalDistance( i_focalDistance )
    {
        // Compute the viewport height from the vertical field of view.
        float verticalFovRadians = gm::Radians( i_verticalFov );

        // This is the important trig ratio which will allow us to compute the viewport height
        // from the focal length.
        float halfViewportHeightOverFocalLength = tan( verticalFovRadians / 2 );
        m_viewportHeight                        = 2.0f * halfViewportHeightOverFocalLength;

        // Compute the width of the virtual viewport.
        m_viewportWidth = m_aspectRatio * m_viewportHeight;

        _UpdateCameraToWorld();
    }

    //-------------------------------------------------------------------------
    /// \name Viewport plane.
    //-------------------------------------------------------------------------

    /// Get the 3D vector matching the virtual viewport width.
    ///
    /// \return Vertical vector of the viewport width.
    inline gm::Vec3f ViewportHorizontal() const
    {
        return m_focalDistance * m_viewportWidth * gm::Vec3f( 1, 0, 0 );
    }

    /// Get the 3D vector matching the virtual viewport height.
    ///
    /// \return Vertical vector of the viewport height.
    inline gm::Vec3f ViewportVertical() const
    {
        return m_focalDistance * m_viewportHeight * gm::Vec3f( 0, 1, 0 );
    }

    /// Get the 3D coordinate of the bottom left corner of the viewport plane.
    ///
    /// \return The bottom left coordinate of the viewport plane.
    inline gm::Vec3f ViewportBottomLeft() const
    {
        return m_focalDistance * gm::Vec3f( 0, 0, 1 ) - ( ViewportHorizontal() * 0.5 ) - ( ViewportVertical() * 0.5 );
    }

    //-------------------------------------------------------------------------
    /// \name LookAtCamera transform (position).
    //-------------------------------------------------------------------------

    /// Get the \em origin, or \em eye of the camera.
    ///
    /// The origin is where rays originate.
    ///
    /// \return the origin of the camera.
    inline const gm::Vec3f& GetOrigin() const
    {
        return m_origin;
    }

    inline void SetOrigin( const gm::Vec3f& i_origin )
    {
        m_origin = i_origin;
        _UpdateCameraToWorld();
    }

private:
    void _UpdateCameraToWorld()
    {
        SetCameraToWorld( gm::LookAt( m_origin, m_lookAt, m_up ) );
    }

    // Look at position & vectors.
    gm::Vec3f m_origin;
    gm::Vec3f m_lookAt;
    gm::Vec3f m_up;

    // The ratio of the width to the height of the image.
    float m_aspectRatio = 0.0f;

    // The fixed height of the virtual viewport.
    float m_viewportHeight = 2.0f;

    // Variable width of the virtual viewport.
    float m_viewportWidth = 0.0f;

    // The distance between the camera origin and the focal plane.
    float m_focalDistance = 1.0f;
};

TOY_NS_CLOSE

