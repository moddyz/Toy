#pragma once

/// \file tri/tri.h
///
/// Tri API
///
/// This header provides all the API entry points of to use the Tri
/// software graphics pipeline.

/// \enum TriDevice
///
/// The device used to execute the graphics commands.
enum class TriDevice : char
{
    CPU = 0,
    CUDA
};

/// \class TriContext
///
/// Storage of top-level state and represents an instance of a graphics
/// processing system.
class TriContext
{
    /// Constructor with default options.
    TriContext( TriDevic );

private:
};

/// \class TriDrawParams
///
/// Root-level parameters for drawing.
class TriDrawParams
{
public:
    /// Set the transformation matrix which moves an object from
    /// camera-space into world-space.
    ///
    /// Also known as the 'camera' transform.
    ///
    /// \param i_cameraToWorld The camera-to-world transformation matrix.
    void SetCameraTransform( const gm::Mat4f& i_cameraToWorld );

    /// Set the transformation matrix which moves objects from camera-space
    /// into the clipping space
    ///
    /// Also known as the 'projection' transformation.
    ///
    /// \param i_cameraToWorld The camera-to-clip transformation matrix.
    void SetProjectionTransform( const gm::Mat4f& i_cameraToClip );

    /// Set the transform which moves objects from clip space into viewport space.
    ///
    /// Also known as the "viewport" transformation.
    ///
    /// \param i_clipToViewport The camera-to-clip transformation matrix.
    void SetViewportTransform( const gm::Mat4f& i_clipToViewport );

private:
    gm::Mat4f m_worldToCamera;
};

