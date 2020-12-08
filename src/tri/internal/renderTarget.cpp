#include "renderTarget.h"
#include "renderBufferCpu.h"
#include "renderBufferCuda.h"
#include "context.h"
#include "bufferUtils.h"
#include "objectContainer.h"

#include <cassert>

/// \var s_renderTargets
///
/// Global internal container of Tri_RenderTarget objects.
///
/// TODO: Exert explicit control over lifetime of this container.
static Tri_ObjectContainer<Tri_RenderTarget> s_renderTargets;

TriStatus
Tri_RenderTargetCreate(TriRenderTarget& target,
                       const Tri_Context* context,
                       int width,
                       int height)
{

    // Create buffers.
    TriBuffer colorBuffer;
    TriStatus status;
    if (context->device == TriDevice_CPU) {
        status = Tri_RenderBufferCreateCPU(
            colorBuffer, context, width, height, TriFormat_Float32_Vec4);
    } else {
        assert(context->device == TriDevice_CUDA);
        status = Tri_RenderBufferCreateCUDA(
            colorBuffer, context, width, height, TriFormat_Float32_Vec4);
    }

    if (status != TriStatus_Success) {
        return status;
    }

    // Allocate internal render target representation.
    typename decltype(s_renderTargets)::EntryT entry =
        s_renderTargets.Create<Tri_RenderTarget>();
    Tri_RenderTarget* renderTarget = entry.second;
    renderTarget->buffers.insert(std::make_pair("color", colorBuffer));
    renderTarget->context = context;

    // Populate opaque object ID.
    target.id = entry.first;

    return TriStatus_Success;
}

TriStatus
Tri_RenderTargetDestroy(TriRenderTarget& target)
{
    Tri_RenderTarget* renderTarget = s_renderTargets.Get(target.id);
    if (renderTarget == nullptr) {
        return TriStatus_RenderTargetNotFound;
    }

    if (renderTarget->context->device == TriDevice_CPU) {
        for (auto pair = renderTarget->buffers.begin();
             pair != renderTarget->buffers.end();
             ++pair) {
            TriBuffer buffer = pair->second;
            Tri_RenderBufferDestroyCPU(buffer);
        }
    } else {
        assert(renderTarget->context->device == TriDevice_CUDA);
        for (auto pair = renderTarget->buffers.begin();
             pair != renderTarget->buffers.end();
             ++pair) {
            TriBuffer buffer = pair->second;
            Tri_RenderBufferDestroyCUDA(buffer);
        }
    }

    bool deleted = s_renderTargets.Delete(target.id);
    assert(deleted);
    target = TriRenderTarget();
    return TriStatus_Success;
}
