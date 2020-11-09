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
/// TriFrameBufferCreate( ctx, frameBuffer, 640, 480 );
///
/// // Allocate graphics pipeline.
/// TriGraphicsPipeline pipeline;
/// TriGraphisPipelineCreate( ctx, pipeline, 640, 480 );
///
/// // Define vertex and indices.
/// gm::Vec3f positions[3] = { ... };
/// uint32_t indices[3] = { 0, 1, 2 };
///
/// // Setup geometry input.
/// TriGeometryInput geoInput;
///
/// geoInput.positions.ptr = static_cast< void* >( &positions );
/// geoInput.positions.format = TriFormat::Float32_Vec3;
/// geoInput.positions.numElements = 3;
/// geoInput.positions.device = TriDevice::CPU;
///
/// geoInput.positions.ptr = static_cast< void* >( &positions );
/// geoInput.positions.format = TriFormat::Float32_Vec3;
/// geoInput.positions.numElements = 3;
/// geoInput.positions.device = TriDevice::CPU;
///
/// // Initialize graphics state.
/// TriGraphicsState graphicsState;
/// state.cameraTransform = ...;
/// state.projectionTransform = ...;
/// state.viewportSize = gm::Vec2i( 640, 480 );
///
/// // Execute graphics pipeline and draw into frame buffer.
/// TriGraphicsPipelineExecute(pipeline, graphicsState, geoInput, frameBuffer);
///
/// // Teardown.
/// TriFrameBufferDestroy( frameBuffer );
/// TriGraphicsPipelineDestroy( pipeline );
/// TriContextDestroy( ctx );
/// \endcode

/// \typedef TriHandle
///
/// User handle to an opaque object.
struct TriHandle
{
    int32_t id;
};

/// \enum TriResult
///
/// API status codes.
enum class TriResult : char
{
    Success = 0,
    ObjectNotFound,
};

/// \struct TriContext
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
/// TriContext preferredCtx;
/// assert(TriContextCreatePreferred(preferredCtx) == TriStatus::Success);
/// assert(preferredCtx.device == TriDevice::CUDA);
///
/// // Construct a context with CPU runtime device selection.
/// TriContext cpuCtx;
/// TriContextCreate(cpuCtx, TriDevice::CPU);
/// assert(cpuCtx.device == TriDevice::CPU);
///
/// /* ... execute graphics ... */
///
/// // Clean-up.
/// TriContextDestroy( cpuCtx );
/// TriContextDestroy( preferredCtx );
/// \endcode
struct TriContext : final
{
    TriId id = TriInvalidId;
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

/// \struct TriBuffer
///
/// A non-ownership-assuming description of a block of memory.
///
/// This is used to transport array-like data for consumption by the graphics
/// pipeline.
struct TriBuffer : final
{
public:
    /// The format of the data.
    TriFormat format{ TriFormat::Invalid };

    /// The current device where the buffer memory resides.
    TriDevice device{ TriDevice::CPU };

    /// Number of elements in the buffer.
    size_t numElements{ 0 };

    /// Pointer to the starting address of the buffer.
    void* ptr{ nullptr };
};

/// \struct TriGeometryInput
///
/// Geometry data serving as input to the graphics pipeline.
struct TriGeometryInput : final
{
public:
    TriBuffer positions;
    TriBuffer indices;
};

/// \struct TriGraphicsState
///
/// Root-level parameters for drawing.
struct TriGraphicsState : final
{
public:
    gm::Mat4f cameraTransform;
    gm::Mat4f projectionTransform;
    gm::Vec2i viewportSize;
};

/// \struct TriGraphicsPipeline
///
/// An executable pipeline which
struct TriGraphicsPipeline : final
{
    TriId id = TriInvalidId;
};

/// Construct a TriContext object with the preferred runtime device.
///
/// The "preferred" runtime device is selected based on priority and
/// availability, by starting from the largest value of
/// \ref TriDevice (bar TriDevice::Count) and incrementing
/// backwards.
///
/// \return The context object.
TriStatus
TriContextCreatePreferred(TriContext& context);

/// Construct a TriContext for a requested runtime device.
///
/// \note The requested runtime device may not be available, in which the
/// fallback CPU runtime device will be selected.
///
/// \param requestedDevice The runtime device used to execute graphics
/// commands.
///
/// \return The context object.
TriStatus
TriContextCreate(TriContext& context, TriDevice requestedDevice);

/// Destroy a TriContext object.
///
/// \note Once a TriContext is destroyed, existing child objects produced
/// via that context will yield undefined behavior.
///
/// \param context The context object to destroy.
TriStatus
TriContextDestroy(TriContext& context);

/// Create a frame buffer to draw into.
///
/// \note A frame buffer allocated with this function must be deallocated
/// via \TriFrameBufferDestroy
///
/// \param buffer The buffer to fill with.
/// \param context The associated context.
/// \param width Pixel width of the frame buffer.
/// \param height Pixel height of the frame buffer.
///
/// \return Allocated frame buffer.
TriStatus
TriFrameBufferCreate(TriBuffer& buffer, const TriContext& context, int width, int height);

/// Destroy an existing frame buffer.
///
/// \param
TriStatus
TriFrameBufferDestroy(TriBuffer& buffer);
