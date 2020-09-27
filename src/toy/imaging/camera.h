#pragma once

/// \file imaging/camera.h
///
/// Base camera imaging.

#include <toy/toy.h>
#include <toy/utils/hostDevice.h>

#include <gm/types/mat4f.h>

TOY_NS_OPEN

/// \class Camera
///
/// The base camera imaging, providing common functionality.
class Camera
{
public:
    /// Default constructor, initializing the cameraToWorld matrix to identity.
    Camera() = default;

    TOY_HOST_DEVICE inline explicit Camera( const gm::Mat4f& i_cameraToWorld )
        : m_cameraToWorld( i_cameraToWorld )
    {
    }

    /// \return The transformation matrix which translates and orients the camera in worldspace.
    TOY_HOST_DEVICE inline const gm::Mat4f& GetCameraToWorld() const
    {
        return m_cameraToWorld;
    }

    /// Set the camera-to-world matrix.
    ///
    /// \param i_cameraToWorld The camera-to-world matrix.
    TOY_HOST_DEVICE inline void SetCameraToWorld( const gm::Mat4f& i_cameraToWorld )
    {
        m_cameraToWorld = i_cameraToWorld;
    }

private:
    // Cached camera-to-world matrix.
    gm::Mat4f m_cameraToWorld = gm::Mat4f::Identity();
};

TOY_NS_CLOSE
