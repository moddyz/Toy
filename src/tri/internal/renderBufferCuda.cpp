#include "renderBufferCuda.h"
#include "bufferUtils.h"

#include <cuda_runtime.h>

TriStatus
Tri_RenderBufferCreateCUDA(TriBuffer& buffer,
                           int width,
                           int height,
                           TriFormat format)
{
    size_t numElements = width * height;
    size_t numBytes = Tri_BufferComputeNumBytes(numElements, format);
    void* bufferPtr;
    cudaError_t err = cudaMalloc(&bufferPtr, numBytes);
    if (bufferPtr != nullptr) {
        // Populate cpu buffer info.
        buffer.ptr = bufferPtr;
        buffer.numElements = numElements;
        buffer.device = TriDevice_CUDA;
        buffer.format = format;
        return TriStatus_Success;
    } else {
        return TriStatus_OutOfMemory;
    }
}

TriStatus
Tri_RenderBuffersCreateCUDA(TriRenderBuffers& buffers, int width, int height)
{
    return Tri_RenderBufferCreateCUDA(
        buffers.color, width, height, TriFormat_Float32_Vec4);
}
