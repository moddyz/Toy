#include "renderer.h"
#include "context.h"
#include "objectContainer.h"

/// \var s_renderers
///
/// Global internal container of Tri_Context objects.
///
/// TODO: Exert explicit control over lifetime of this container.
static Tri_ObjectContainer<Tri_Renderer> s_renderers;

TriStatus
Tri_RendererCreate(TriRenderer& renderer, const Tri_Context* context)
{
    // Allocate new internal context object.
    typename decltype(s_renderers)::EntryT entry =
        s_renderers.Create<Tri_Renderer>();
    entry.second->context = context;

    // Populate opaque object ID.
    renderer.id = entry.first;

    return TriStatus_Success;
}
