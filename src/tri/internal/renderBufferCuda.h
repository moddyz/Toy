#pragma once

#include <tri/tri.h>

// Internal details for CUDA frame buffer management.

// Allocate a single CUDA render buffer.
TriStatus
Tri_RenderBufferCreateCUDA(TriBuffer& buffer, int width, int height, TriFormat format);

// Allocate CUDA render buffers.
TriStatus
Tri_RenderBuffersCreateCUDA(TriRenderBuffers& buffers, int width, int height);
