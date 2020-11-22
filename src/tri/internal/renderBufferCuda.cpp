#include "renderBufferCuda.h"
#include "bufferUtils.h"

#include <cuda_runtime.h>
#include <cassert>

TriStatus
Tri_RenderBufferCreateCUDA(TriBuffer& buffer,
                           const Tri_Context* context,
                           int width,
                           int height,
                           TriFormat format)
{
    size_t numElements = width * height;
    size_t numBytes = Tri_BufferComputeNumBytes(numElements, format);
    void* bufferPtr{ nullptr };
    cudaError_t err = cudaMalloc(&bufferPtr, numBytes);
    if (err == cudaSuccess) {
        Tri_BufferCreate(
            buffer, context, numElements, format, TriDevice_CUDA, bufferPtr);
        return TriStatus_Success;
    } else {
        return TriStatus_OutOfMemory;
    }
}

TriStatus
Tri_RenderBufferDestroyCUDA(TriBuffer& buffer)
{
    Tri_Buffer* internalBuffer = Tri_BufferGet(buffer.id);
    if (internalBuffer == nullptr) {
        return TriStatus_ContextNotFound;
    }

    // Deallocate memory.
    assert(internalBuffer->ptr != nullptr);
    cudaFree(internalBuffer->ptr);

    // Unregister internal buffer information.
    Tri_BufferDelete(buffer);

    return TriStatus_Success;
}
