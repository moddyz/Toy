// Implementation of public API.
// All symbols with Tri_ prefix are internal.

#include "tri.h"

#include "internal/context.h"
#include "internal/renderer.h"
#include "internal/renderBuffers.h"

#include <cassert>

TriStatus
TriContextCreatePreferred(TriContext& context)
{
    TriDevice device = Tri_SelectPreferredDevice();
    return Tri_ContextCreate(context, device);
}

TriStatus
TriContextCreate(TriContext& context, TriDevice requestedDevice)
{
    // Check for device availability.
    if (requestedDevice == TriDevice_CUDA && !Tri_IsCUDASupported()) {
        return TriStatus_UnsupportedDevice;
    }

    return Tri_ContextCreate(context, requestedDevice);
}

TriStatus
TriContextDestroy(TriContext& context)
{
    return Tri_ContextDestroy(context);
}

TriStatus
TriContextGetDevice(const TriContext& context, TriDevice& device)
{
    Tri_Context* ctx = Tri_ContextGet(context.id);
    if (ctx == nullptr) {
        return TriStatus_ContextNotFound;
    }

    device = ctx->device;
    return TriStatus_Success;
}

TriStatus
TriRendererCreate(TriRenderer& renderer, const TriContext& context)
{
    Tri_Context* ctx = Tri_ContextGet(context.id);
    if (ctx == nullptr) {
        return TriStatus_ContextNotFound;
    }

    return Tri_RendererCreate(renderer, ctx);
}

TriStatus
TriRendererDestroy(TriRenderer& renderer)
{
    return Tri_RendererDestroy(renderer);
}

TriStatus
TriRendererSetCameraXform(TriRenderer& renderer, float* cameraXform)
{
    Tri_Renderer* internalRenderer = Tri_RendererGet(renderer.id);
    if (internalRenderer == nullptr) {
        return TriStatus_RendererNotFound;
    }

    memcpy(&(internalRenderer->cameraXform), cameraXform, sizeof(float) * 16);
    return TriStatus_Success;
}

TriStatus
TriRendererSetProjectionXform(TriRenderer& renderer, float* projectionXform)
{
    Tri_Renderer* internalRenderer = Tri_RendererGet(renderer.id);
    if (internalRenderer == nullptr) {
        return TriStatus_RendererNotFound;
    }

    memcpy(&(internalRenderer->projectionXform),
           projectionXform,
           sizeof(float) * 16);
    return TriStatus_Success;
}

TriStatus
TriRenderBuffersCreate(TriRenderBuffers& buffers,
                       const TriContext& context,
                       int width,
                       int height)
{
    Tri_Context* ctx = Tri_ContextGet(context.id);
    if (ctx == nullptr) {
        return TriStatus_ContextNotFound;
    }

    return Tri_RenderBuffersCreate(buffers, ctx, width, height);
}

TriStatus
TriRenderBuffersDestroy(TriRenderBuffers& buffers)
{
    return Tri_RenderBuffersDestroy(buffers);
}
