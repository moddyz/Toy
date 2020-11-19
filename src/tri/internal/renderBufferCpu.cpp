#include "renderBufferCpu.h"
#include "bufferUtils.h"

#include <stdlib.h>

TriStatus
Tri_RenderBufferCreateCPU(TriBuffer& buffer,
                          int width,
                          int height,
                          TriFormat format)
{
    size_t numElements = width * height;
    size_t numBytes = Tri_BufferComputeNumBytes(numElements, format);
    void* bufferPtr = malloc(numBytes);
    if (bufferPtr != nullptr) {
        // Populate cpu buffer info.
        buffer.ptr = bufferPtr;
        buffer.numElements = numElements;
        buffer.device = TriDevice_CPU;
        buffer.format = format;
        return TriStatus_Success;
    } else {
        return TriStatus_OutOfMemory;
    }
}

TriStatus
Tri_RenderBuffersCreateCPU(TriRenderBuffers& buffers, int width, int height);
{
    return Tri_RenderBufferCreateCPU(buffers.color, width, height);
}
