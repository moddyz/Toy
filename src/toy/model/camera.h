#pragma once

/// \file model/camera.h
///
/// Base camera model.

#include <toy/toy.h>
#include <toy/utils/hostDevice.h>

#include <gm/types/mat4f.h>

TOY_NS_OPEN

/// \class Camera
///
/// The base camera model, providing common functionality.
class Camera
{
public:
    inline explicit Camera( const gm::Mat4f& i_cameraToWorld )
        : m_cameraToWorld( i_cameraToWorld )
    {
    }

    /// \return The transformation matrix which translates and orients the camera in worldspace.
    inline const gm::Mat4f& GetCameraToWorld() const
    {
        return m_cameraToWorld;
    }

    /// Set the camera-to-world matrix.
    ///
    /// \param i_cameraToWorld The camera-to-world matrix.
    inline void SetCameraToWorld( const gm::Mat4f& i_cameraToWorld ) const
    {
        m_cameraToWorld = i_cameraToWorld;
    }

private:
    //
    gm::Mat4f m_cameraToWorld;
};

TOY_NS_CLOSE
