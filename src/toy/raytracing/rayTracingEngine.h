#pragma once

#include <toy/engine/renderEngine.h>
#include <toy/engine/renderSettings.h>
#include <toy/memory/residency.h>
#include <toy/toy.h>

TOY_NS_OPEN

/// \class RayTracingEngine
///
/// Integration of imagings and operators to render an image, using the \em ray-tracing technique.
template < Residency ResidencyT >
class RayTracingEngine : public RenderEngine< ResidencyT >
{
public:
    virtual bool Render( const Scene& i_scene, Image& o_image ) override;

protected:
    // Allocate memory and data structures for rendering.
    virtual bool _Initialize() override;
};

TOY_NS_CLOSE
