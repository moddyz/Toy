#pragma once

#include <tri/tri.h>

class Tri_Context;

/// \class Tri_Renderer
///
/// Internal TriContext representation.
struct Tri_Renderer
{
    // Associated context.
    const Tri_Context* context{ nullptr };
};

/// Device agnostic method for creating a renderer.
///
/// \param renderer The opaque object handle to the renderer.
/// \param context The owning context.
///
/// \retval TriStatus_Success Successful creation of the renderer.
TriStatus
Tri_RendererCreate(TriRenderer& renderer, const Tri_Context* context);
