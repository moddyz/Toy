#pragma once

// Internal functionality for managing render buffers.

#include <tri/tri.h>

class Tri_Context;

// Create render buffers for associated context.
TriStatus
Tri_RenderBuffersCreate(TriRenderBuffers& buffers,
                        const Tri_Context* context,
                        int width,
                        int height);

// Destroy render buffers.
TriStatus
Tri_RenderBuffersDestroy(TriRenderBuffers& buffers);
