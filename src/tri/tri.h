#pragma once

/// \file tri/tri.h
///
/// Tri API
///
/// \b Tri is a software renderer implementing a fixed rasterization pipeline.
///
/// This header provides all the API entry points of the Tri renderer.

/// \class TriContext
///
/// Describes root-level properties of a Tri renderer.
///
/// After the context is initialized with specified properties, those
/// properties are \em immutable for the lifetime of the context object.
///
/// The client must construct a new context if different property
/// specifications are desired.
///
/// Example usage:
/// \code{.cpp}
/// // Expected behavior when running on a machine with CUDA runtime support.
///
/// // Preferred context.
/// TriContext preferredCtx = TriContextCreatePreferred();
/// assert(preferredCtx.device == TriDevice::CUDA);
///
/// // Construct a context with CPU runtime device selection.
/// TriContext cpuCtx = TriContextCreate(TriDevice::CPU);
/// assert(cpuCtx.device == TriDevice::CPU);
///
/// /* ... execute graphics ... */
///
/// // Clean-up.
/// TriContextDestroy( cpuCtx );
/// TriContextDestroy( preferredCtx );
/// \endcode
class TriContext : final
{
public:
    TriDevice device;
};

/// \enum TriDevice
///
/// The device used to execute the graphics operations.
enum class TriDevice : char
{
    CPU = 0,
    CUDA,
    Count
};

/// \enum TriFormat
///
/// The elemental format of data structures.
enum class TriFormat : char
{
    Invalid = 0,
    Float32,
    Float32_Vec2,
    Float32_Vec3,
    Float32_Vec4,
    Uint32,
    Uint8_Vec3,
    Uint8_Vec4,
    Count
};

/// \class TriBuffer
///
/// A non-ownership-assuming description of a block of memory.
///
/// This is used to transport array-like data for consumption by the graphics
/// pipeline.
class TriBuffer : final
{
public:
    /// The format of the data.
    TriFormat format{ TriFormat::Invalid };

    /// The current device where the buffer memory resides.
    TriDevice device{ TriDevice::CPU };

    /// Number of elements in the buffer.
    size_t numElements = 0;

    /// Pointer to the starting address of the buffer.
    void* ptr = nullptr;
};

/// \class TriGeometryInput
///
/// Geometry data serving as input to the graphics pipeline.
class TriGeometryInput : final
{
public:
    TriBuffer positions;
    TriBuffer indices;
};

/// \class TriGraphicsState
///
/// Root-level parameters for drawing.
class TriGraphicsState : final
{
public:
    gm::Mat4f cameraTransform;
    gm::Mat4f projectionTransform;
    gm::Vec2i viewportSize;
};

/// \class TriGraphicsPipeline
///
/// An executable pipeline which
class TriGraphicsPipeline
{
private:
    void* m_impl = nullptr;
};

/// Construct a TriContext object with the preferred runtime device.
///
/// The "preferred" runtime device is selected based on priority and
/// availability, by starting from the largest value of \ref
/// TriDevice (bar TriDevice::Count) and incrementing
/// backwards.
///
/// \return The context object.
TriContext
TriContextCreatePreferred();

/// Construct a TriContext for a requested runtime device.
///
/// \note The requested runtime device may not be available, in which the
/// fallback CPU runtime device will be selected.
///
/// \param i_requestedDevice The runtime device used to execute graphics
/// commands.
///
/// \return The context object.
TriContext
TriContextCreate(TriDevice i_requestedDevice);

/// Destroy a TriContext object.
///
/// \note Once a TriContext is destroyed, existing child objects produced
/// via that context will yield undefined behavior.
///
/// \param o_context The context object to destroy.
bool
TriContextDestroy(TriContext& o_context);

/// Allocate a frame buffer to draw into.
///
/// \note A frame buffer allocated with this function must be deallocated
/// via \TriFrameBufferDestroy
TriBuffer
TriFrameBufferCreate(const TriContext& i_context, int width, int height);

/// Deallocate a frame buffer.
bool
TriFrameBufferDestroy(TriBuffer& o_buffer);
