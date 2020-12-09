#pragma once

// Internal functionality for managing render target.

#include <tri/tri.h>

#include <unordered_map>

class Tri_Context;

/// \class Tri_RenderTarget
///
/// Internal render target representation.
///
/// Stores buffers used as the output of a render.
struct Tri_RenderTarget
{
    /// Buffers associated with this render target.
    std::unordered_map<TriToken, TriBuffer> buffers;

    /// Associated context.
    const Tri_Context* context{ nullptr };
};

// Create render target for associated context.
TriStatus
Tri_RenderTargetCreate(TriRenderTarget& target,
                       const Tri_Context* context,
                       int width,
                       int height);

// Query one of the buffers of the render target.
TriStatus
Tri_RenderTargetBuffer(const TriRenderTarget& target,
                       const TriToken& name,
                       TriBuffer& buffer);

// Destroy render target.
TriStatus
Tri_RenderTargetDestroy(TriRenderTarget& target);
