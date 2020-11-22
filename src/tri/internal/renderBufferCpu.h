#pragma once

// Internal details for CPU frame buffer management.

#include <tri/tri.h>

// Forward declaration.
class Tri_Context;

// Allocate a single CPU render buffer.
TriStatus
Tri_RenderBufferCreateCPU(TriBuffer& buffer,
                          const Tri_Context* context,
                          int width,
                          int height,
                          TriFormat format);

// Deallocate a single CPU render buffer.
TriStatus
Tri_RenderBufferDestroyCPU(TriBuffer& buffer);
