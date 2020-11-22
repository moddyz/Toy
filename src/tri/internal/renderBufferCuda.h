#pragma once

// Internal details for CUDA frame buffer management.

#include <tri/tri.h>

// Forward declaration.
class Tri_Context;

// Allocate a single CUDA render buffer.
TriStatus
Tri_RenderBufferCreateCUDA(TriBuffer& buffer,
                           const Tri_Context* context,
                           int width,
                           int height,
                           TriFormat format);

// Deallocate a single CUDA render buffer.
TriStatus
Tri_RenderBufferDestroyCUDA(TriBuffer& buffer);
