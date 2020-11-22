#include "renderBuffers.h"
#include "renderBufferCpu.h"
#include "renderBufferCuda.h"
#include "context.h"
#include "bufferUtils.h"

#include <cassert>

TriStatus
Tri_RenderBuffersCreate(TriRenderBuffers& buffers,
                        const Tri_Context* context,
                        int width,
                        int height)
{
    if (context->device == TriDevice_CPU) {
        return Tri_RenderBufferCreateCPU(
            buffers.color, context, width, height, TriFormat_Float32_Vec4);
    } else {
        assert(context->device == TriDevice_CUDA);
        return Tri_RenderBufferCreateCUDA(
            buffers.color, context, width, height, TriFormat_Float32_Vec4);
    }
}

TriStatus
Tri_RenderBuffersDestroy(TriRenderBuffers& buffers)
{
    Tri_Buffer* internalBuffer = Tri_BufferGet(buffers.color.id);
    if (internalBuffer == nullptr) {
        return TriStatus_ObjectNotFound;
    }

    if (internalBuffer->device == TriDevice_CPU) {
        return Tri_RenderBufferDestroyCPU(buffers.color);
    } else {
        assert(internalBuffer->device == TriDevice_CUDA);
        return Tri_RenderBufferDestroyCUDA(buffers.color);
    }
}
