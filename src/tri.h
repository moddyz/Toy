#pragma once

/// \file tri/tri.h
///
/// Tri API
///
/// \b Tri is a software-based rasterization-based renderer.
///
/// This header provides all the API entry points of for producing graphics with
/// the Tri renderer.

/// \enum TriDevice
///
/// The device used to execute the graphics commands.
enum class TriDevice : char
{
    CPU = 0,
    CUDA,
    Count
};

/// \class TriContext
///
/// Storage of top-level state and represents an instance of a graphics
/// processing system.
///
/// Example usage:
/// \code{.cpp}
/// // Expected behavior when running on a machine with CUDA support.
///
/// // Preferred context.
/// TriContext preferredCtx;
/// assert(preferredCtx.GetDevice() == TriDevice::CUDA);
///
/// // Construct a context with CPU device selection.
/// TriContext cpuCtx(TriDevice::CPU);
/// assert(cudaCtx.GetDevice() == TriDevice::CPU);
/// \endcode
class TriContext : final
{
public:
    /// Construct a TriContext with the preferred device.
    ///
    /// The "preferred" device is selected based on priority and availability,
    /// by starting from the largest value of \ref TriDevice (bar Count)
    /// and incrementing backwards.
    TriContext();

    /// Construct a TriContext for a requested device.
    ///
    /// \note The requested device may not be available, in which the fallback
    /// CPU device will be selected.
    ///
    /// \param i_device The device used to execute graphics commands.
    explicit TriContext(TriDevice i_requestedDevice);

    /// Get the actual device selected by this context.
    ///
    /// \return The device used to execute graphics commands.
    inline TriDevice GetDevice() const { return m_device; }

private:
    TriDevice m_device;
};

/// \class TriDrawParams
///
/// Root-level parameters for drawing.
class TriDrawParams : final
{
public:
    /// Set the transformation matrix which moves an object from/ camera-space
    /// into world-space.
    ///
    /// Also known as the 'camera' transform.
    ///
    /// \note There is a computation cost in setting this transform - as the
    /// matrix will need to be \em inverted before it is successfully stored.
    ///
    /// \param i_cameraTransform The camera-to-world transformation matrix.
    ///
    /// \return If the transformation was successfully set (the matrix is
    /// invertable).
    bool SetCameraTransform(const gm::Mat4f& i_cameraTransform);

    /// Set the transformation matrix which moves objects from camera-space
    /// into the clipping space.
    ///
    /// Also known as the 'projection' transformation.
    ///
    /// \param i_projectionTransform The camera-to-clip transformation matrix.
    void SetProjectionTransform(const gm::Mat4f& i_projectionTransform);

    /// Set the transform which moves objects from clip space into viewport
    /// space.
    ///
    /// Also known as the "viewport" transformation.
    ///
    /// \param i_viewportTransform The camera-to-clip transformation matrix.
    void SetViewportTransform(const gm::Mat4f& i_viewportTransform);

private:
    gm::Mat4f m_cameraTransform;
    gm::Mat4f m_projectionTransform;
    gm::Mat4f m_viewportTransform;
};

