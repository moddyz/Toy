#pragma once

#include <tri/tri.h>

#include <gm/types/mat4f.h>

class Tri_Context;

/// \class Tri_Renderer
///
/// Internal TriContext representation.
struct Tri_Renderer
{
    // Associated context.
    const Tri_Context* context{ nullptr };

    // Renderer state.
    gm::Mat4f cameraXform{ gm::Mat4f::Identity() };
    gm::Mat4f projectionXform{ gm::Mat4f::Identity() };
    gm::Mat4f viewportXform{ gm::Mat4f::Identity() };
};

/// Device agnostic method for creating a renderer.
///
/// \param renderer The opaque object handle to the renderer.
/// \param context The owning context.
///
/// \retval TriStatus_Success Successful creation of the renderer.
TriStatus
Tri_RendererCreate(TriRenderer& renderer, const Tri_Context* context);

/// Query the internal renderer representation.
Tri_Renderer*
Tri_RendererGet(TriId id);

/// Delete a renderer.
TriStatus
Tri_RendererDestroy(TriRenderer& renderer);
