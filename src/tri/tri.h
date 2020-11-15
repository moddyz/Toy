#pragma once

/// \file tri/tri.h
///
/// Tri API
///
/// \b Tri is a software renderer implementing a fixed rasterization pipeline.
///
/// This header provides all the API entry points of the Tri renderer.
///
/// Example of rendering a triangle:
/// \code{.cpp}
/// // Initialize context.
/// TriContext ctx;
/// TriCreateContextPreferred(ctx);
///
/// // Allocate frame buffer.
/// TriBuffer frameBuffer;
/// TriFrameBufferCreate(ctx, frameBuffer, 640, 480);
///
/// // Allocate graphics pipeline.
/// TriPipeline pipeline;
/// TriPipelineCreate(ctx, pipeline);
///
/// // Define vertex and indices.
/// gm::Vec3f positions[3] = { ... };
/// uint32_t indices[3] = { 0, 1, 2 };
///
/// // Setup geometry input.
/// TriGeometryInput geoInput;
///
/// geoInput.positions.ptr = static_cast< void* >( &positions );
/// geoInput.positions.format = TriFormat_Float32_Vec3;
/// geoInput.positions.numElements = 3;
/// geoInput.positions.device = TriDevice::CPU;
///
/// geoInput.indices.ptr = static_cast< void* >( &indices );
/// geoInput.indices.format = TriFormat_Uint32;
/// geoInput.indices.numElements = 3;
/// geoInput.indices.device = TriDevice::CPU;
///
/// // Set rendering properties.
/// TriPipelineSetCameraXform(pipeline, gm::Mat4f::Identity());
/// TriPipelineSetProjectionXform(pipeline, gm::Mat4f::Identity());
///
/// // Execute graphics pipeline and draw into frame buffer.
/// TriPipelineExecute(pipeline, geoInput, frameBuffer);
///
/// // Teardown.
/// TriFrameBufferDestroy(frameBuffer);
/// TriPipelineDestroy(pipeline);
/// TriContextDestroy(ctx);
/// \endcode

#include <stdint.h>
#include <stddef.h>

/// \typedef TriId
///
/// Type definition for a Tri opaque object identifier.
using TriId = int32_t;

/// \var TriId_Invalid
///
/// Constant variable of an invalid object ID.
constexpr TriId TriId_Invalid = -1;

/// \enum TriStatus
///
/// API return codes.
enum TriStatus
{
    /// API call returned with no errors.
    TriStatus_Success = 0,

    /// This error is returned from query APIs where the object associated
    /// with the opaque object ID cannot be found.
    TriStatus_ObjectNotFound,

    /// This error is returned when querying a property not associated with
    /// the object type.
    TriStatus_InvalidProperty
};

/// \enum TriDevice
///
/// The device used to execute the graphics operations.
enum TriDevice
{
    TriDevice_Invalid = 0,
    TriDevice_CPU,
    TriDevice_CUDA,
    TriDevice_Count
};

/// \enum TriFormat
///
/// The elemental format of data structures.
enum TriFormat
{
    TriFormat_Invalid = 0,
    TriFormat_Float32,
    TriFormat_Float32_Vec2,
    TriFormat_Float32_Vec3,
    TriFormat_Float32_Vec4,
    TriFormat_Uint32,
    TriFormat_Uint8_Vec3,
    TriFormat_Uint8_Vec4,
    TriFormat_Count
};

/// \enum TriProperty
///
/// Properties of Tri opaque objects.
enum TriProperty
{
    // TriPipeline.
    TriProperty_CameraXform = 0,
    TriProperty_ProjectionXform,
    TriProperty_RenderTargetSize
};

/// \struct TriContext
///
/// Root-level opaque object, specifying the device and lifetime of
/// internal memory structures.
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
/// TriContext preferredCtx;
/// assert(TriContextCreatePreferred(preferredCtx) == TriStatus::Success);
/// assert(preferredCtx.device == TriDevice_CUDA);
///
/// // Construct a context with CPU runtime device selection.
/// TriContext cpuCtx;
/// TriContextCreate(cpuCtx, TriDevice_CPU);
/// assert(cpuCtx.device == TriDevice_CPU);
///
/// /* ... execute graphics ... */
///
/// // Clean-up.
/// TriContextDestroy( cpuCtx );
/// TriContextDestroy( preferredCtx );
/// \endcode
struct TriContext
{
    TriId id{ TriId_Invalid };
};

/// \struct TriPipeline
///
/// An executable pipeline which
struct TriPipeline
{
    TriId id{ TriId_Invalid };
};

/// \struct TriBuffer
///
/// A non-ownership-assuming description of a block of memory.
///
/// This is used to transport array-like data for consumption by the graphics
/// pipeline.
struct TriBuffer
{
public:
    /// The format of the data.
    TriFormat format{ TriFormat_Invalid };

    /// The current device where the buffer memory resides.
    TriDevice device{ TriDevice_Invalid };

    /// Number of elements in the buffer.
    size_t numElements{ 0 };

    /// Pointer to the starting address of the buffer.
    void* ptr{ nullptr };
};

/// \struct TriGeometryInput
///
/// Geometry data serving as input to the graphics pipeline.
struct TriGeometryInput
{
public:
    TriBuffer positions;
    TriBuffer indices;
};

/// Create a TriContext object with the preferred runtime device.
///
/// The "preferred" runtime device is selected based on priority and
/// availability, by starting from the largest value of
/// \ref TriDevice (bar TriDevice::Count) and incrementing
/// backwards.
///
/// \retval TriStatus_Success Successful creation of a preferred context.
TriStatus
TriContextCreatePreferred(TriContext& context);

/// Construct a TriContext for a requested runtime device.
///
/// \note The requested runtime device may not be available, in which the
/// fallback CPU runtime device will be selected.
///
/// \param requestedDevice Requested device for the context.
///
/// \retval TriStatus_Success Successful creation of a context with requested
/// properties.
TriStatus
TriContextCreate(TriContext& context, TriDevice requestedDevice );

/// Destroy a TriContext object.
///
/// \note Once a TriContext is destroyed, existing child objects produced
/// via that context will yield undefined behavior.
///
/// \param context The context object to destroy.
TriStatus
TriContextDestroy(TriContext& context);

/// Query the \p context's associated device.
///
/// \retval TriStatus_Success Successful device query.
TriStatus
TriContextGetDevice(const TriContext& context, TriDevice& device );

/// Create a RGBA frame buffer.
///
/// \note A frame buffer allocated with this function must be deallocated
/// via \ref TriFrameBufferDestroy
///
/// \param buffer The buffer to fill with.
/// \param context The associated context.
/// \param width Pixel width of the frame buffer.
/// \param height Pixel height of the frame buffer.
///
/// \return Allocated frame buffer.
TriStatus
TriFrameBufferCreate(TriBuffer& buffer,
                     const TriContext& context,
                     int width,
                     int height);

/// Destroy an existing frame buffer.
///
/// \param
TriStatus
TriFrameBufferDestroy(TriBuffer& buffer);
