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
/// // Create renderer.
/// TriRenderer renderer;
/// TriRendererCreate(ctx, renderer);
///
/// // Set camera and viewport state.
/// TriRendererSetCameraXform(renderer, gm::Mat4f::Identity());
/// TriRendererSetProjectionXform(renderer, gm::Mat4f::Identity());
/// TriRendererSetViewportSize(renderer, 640, 480);
///
/// // Create render target.
/// TriRenderTarget target;
/// TriRenderTargetCreate(target, ctx, 640, 480);
///
/// // Define vertex and indices.
/// gm::Vec3f positions[3] = { ... };
/// uint32_t indices[3] = { 0, 1, 2 };
///
/// // Setup geometry input.
/// TriGeometryInput geoInput;
///
/// TriBufferMap(
///     geoInput.positions,
///     ctx,
///     static_cast< void* >( &positions ),
///     TriFormat_Float32_Vec3,
///     3,
///     TriDevice_CPU
/// );
///
/// TriBufferMap(
///     geoInput.indices,
///     ctx,
///     static_cast< void* >( &indices ),
///     TriFormat_Uint32,
///     3,
///     TriDevice_CPU
/// );
///
/// // Execute rendering operation and write into buffers.
/// TriRendererExecute(renderer, geoInput, target);
///
/// // Teardown.
/// TriRenderTargetDestroy(target);
/// TriRendererDestroy(renderer);
/// TriContextDestroy(ctx);
/// \endcode

#include <stdint.h>
#include <stddef.h>
#include <string>

/// \typedef TriId
///
/// Type definition for a Tri opaque object identifier.
using TriId = int32_t;

/// \var TriId_Uninitialized
///
/// Constant variable of an invalid object ID.
constexpr TriId TriId_Uninitialized = -1;

/// \typedef TriToken
///
/// Storage of a sequence of characters.
using TriToken = std::string;

/// \enum TriStatus
///
/// API return codes.
enum TriStatus
{
    /// API call returned with no errors.
    TriStatus_Success = 0,

    /// Returned when the requested device is not supported by the runtime
    /// environment.
    TriStatus_UnsupportedDevice = 0,

    /// These error statuses are returned from query APIs where the
    /// object associated with the ID cannot be found.
    TriStatus_ContextNotFound,
    TriStatus_RendererNotFound,
    TriStatus_RenderTargetNotFound,
    TriStatus_BufferNotFound,

    /// This error is returned when attemping to allocate a buffer but the
    /// system has run out of memory capacity.
    TriStatus_OutOfMemory
};

/// \enum TriDevice
///
/// The device used to execute the graphics operations.
enum TriDevice
{
    TriDevice_Uninitialized = 0,
    TriDevice_CPU,
    TriDevice_CUDA,
    TriDevice_Count
};

/// \enum TriFormat
///
/// The elemental format of data structures.
enum TriFormat
{
    TriFormat_Uninitialized = 0,
    TriFormat_Float32,
    TriFormat_Float32_Vec2,
    TriFormat_Float32_Vec3,
    TriFormat_Float32_Vec4,
    TriFormat_Uint32,
    TriFormat_Uint8_Vec3,
    TriFormat_Uint8_Vec4,
    TriFormat_Count
};

/// \struct TriContext
///
/// Root-level opaque object, specifying the device and  allocated
///
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
    TriId id{ TriId_Uninitialized };
};

/// \struct TriRenderer
///
/// A opaque representation of an \em executable renderer.
///
/// Create this object
struct TriRenderer
{
    TriId id{ TriId_Uninitialized };
};

/// \struct TriBuffer
///
/// A opaque representation about a block of memory tracked by the system.
struct TriBuffer
{
    TriId id{ TriId_Uninitialized };
};

/// \struct TriRenderTarget
///
/// An opaque representation of the outputs of a render operation.
struct TriRenderTarget
{
    TriId id{ TriId_Uninitialized };
};

/// \struct TriGeometryInput
///
/// Geometry data serving as input to the graphics renderer.
struct TriGeometryInput
{
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
/// \param context The opaque context object to initialize.
///
/// \pre \p context must be un-initialised (\ref TriId_Uninitialized);
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
/// \retval TriStatus_Success Successful creation.
/// \retval TriStatus_UnsupportedDevice Error status when the requested device
/// is not supported by the runtime environment.
TriStatus
TriContextCreate(TriContext& context, TriDevice requestedDevice);

/// Destroy an initialized TriContext object.
///
/// \note Once a TriContext is destroyed, existing child objects produced
/// via that context will yield undefined behavior.
///
/// \param context The context object to destroy.
///
/// \retval TriStatus_Success Successful destruction.
/// \retval TriStatus_ContextNotFound \p context does not exist.
TriStatus
TriContextDestroy(TriContext& context);

/// Query the \p context's associated device.
///
/// \param context The context to query the device for.
/// \param device Output device variable.
///
/// \retval TriStatus_Success Successful query.
/// \retval TriStatus_ContextNotFound \p context does not exist.
TriStatus
TriContextGetDevice(const TriContext& context, TriDevice& device);

/// Create a renderer.
///
/// \param renderer The opaque object handle to the renderer.
/// \param context The owning context.
///
/// \retval TriStatus_Success Successful creation of the renderer.
/// \retval TriStatus_ContextNotFound \p context does not exist.
TriStatus
TriRendererCreate(TriRenderer& renderer, const TriContext& context);

/// Destroy an existing renderer.
///
/// \param renderer The opaque object handle to the renderer.
///
/// \retval TriStatus_Success Successful destruction of the renderer.
/// \retval TriStatus_RendererNotFound \p renderer is uninitialized.
TriStatus
TriRendererDestroy(TriRenderer& renderer);

/// Set the camera-to-world transform.
///
/// \param renderer The opaque object handle to the renderer.
/// \param cameraXform The 4-by-4 camera-to-world transform.
///
/// \pre \p cameraXform must be 16 floats forming a row-major matrix.
///
/// \retval TriStatus_Success Successful state update.
/// \retval TriStatus_RendererNotFound \p renderer is uninitialized.
TriStatus
TriRendererSetCameraXform(TriRenderer& renderer, float* cameraXform);

/// Set-up the renderer camera with a perspective projection transformation.
///
/// \param renderer The opaque object handle to the renderer.
/// \param verticalFov Vertical angle in degrees of the camera eye.
/// \param aspectRatio The ratio of the width to the height of viewing plane.
/// \param near Distance from the camera eye to the \em near clipping plane.
/// \param far Distance from the camera eye to the \em far clipping plane.
///
/// \retval TriStatus_Success Successful state update.
/// \retval TriStatus_RendererNotFound \p renderer is uninitialized.
TriStatus
TriRendererPerspective(TriRenderer& renderer,
                       float verticalFov,
                       float aspectRatio,
                       float near,
                       float far);

/// Set-up the renderer viewport transformation.
///
/// The viewport transformation maps from NDC (normalized device coordinates)
/// space into the a rectangular region in window space (pixel units).
///
/// \param renderer The opaque object handle to the renderer.
/// \param offsetX  Bottom left corner x-coord of the rectangular region.
/// \param offsetY  Bottom left corner y-coord of the rectangular region.
/// \param width Width of the rectangular region.
/// \param height Height of the viewport.
///
/// \retval TriStatus_Success Successful state update.
/// \retval TriStatus_RendererNotFound \p renderer is uninitialized.
TriStatus
TriRendererViewport(TriRenderer& renderer,
                    float offsetX,
                    float offsetY,
                    float width,
                    float height);

/// Create a render target serving as the output of a rendering operation.
///
/// \note A render buffer allocated with this function must be deallocated
/// via \ref TriRenderTargetDestroy
///
/// \param target The render target.
/// \param context The associated context.
/// \param width Pixel width of the associated 2D buffers.
/// \param height Pixel height of the associated 2D buffers.
///
/// \retval TriStatus_Success Successful allocation of the render target (and underlying buffers).
/// \retval TriStatus_ContextNotFound \p context does not exist.
TriStatus
TriRenderTargetCreate(TriRenderTarget& target,
                      const TriContext& context,
                      int width,
                      int height);

/// Destroy an existing render target.
///
/// \param target The render target to deallocate.
///
/// \retval TriStatus_Success Successful destruction of render target.
/// \retval TriStatus_RenderTargetNotFound \p target is not a valid render target.
TriStatus
TriRenderTargetDestroy(TriRenderTarget& target);
