#include "renderBufferCpu.h"
#include "bufferUtils.h"

#include <stdlib.h>
#include <cassert>

TriStatus
Tri_RenderBufferCreateCPU(TriBuffer& buffer,
                          const Tri_Context* context,
                          int width,
                          int height,
                          TriFormat format)
{
    assert(buffer.id == TriId_Uninitialized);

    size_t numElements = width * height;
    size_t numBytes = Tri_BufferComputeNumBytes(numElements, format);
    void* bufferPtr = malloc(numBytes);
    if (bufferPtr != nullptr) {
        Tri_BufferCreate(
            buffer, context, numElements, format, TriDevice_CPU, bufferPtr);
        return TriStatus_Success;
    } else {
        return TriStatus_OutOfMemory;
    }
}

TriStatus
Tri_RenderBufferDestroyCPU(TriBuffer& buffer)
{
    Tri_Buffer* internalBuffer = Tri_BufferGet(buffer.id);
    if (internalBuffer == nullptr) {
        return TriStatus_ContextNotFound;
    }

    // Deallocate memory.
    assert(internalBuffer->ptr != nullptr);
    free(internalBuffer->ptr);

    // Unregister internal buffer information.
    Tri_BufferDelete(buffer);

    return TriStatus_Success;
}
