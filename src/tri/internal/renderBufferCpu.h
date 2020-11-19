#pragma once

#include <tri/tri.h>

// Internal details for CPU frame buffer management.

// Allocate a single CPU render buffer.
TriStatus
Tri_RenderBufferCreateCPU(TriBuffer& buffer, int width, int height, TriFormat format);

// Allocate CPU render buffers.
TriStatus
Tri_RenderBuffersCreateCPU(TriRenderBuffers& buffers, int width, int height);
