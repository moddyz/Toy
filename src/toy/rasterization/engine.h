#pragma once

#include <toy/memory/residency.h>

#include <toy/imaging/renderEngine.h>
#include <toy/imaging/renderSettings.h>

TOY_NS_OPEN

/// \class RasterizationEngine
///
/// Integration of imagings and operators to render an image, using the \em rasterization technique.
template < Residency ResidencyT >
class RasterizationEngine : public RenderEngine< ResidencyT >
{
public:
    virtual bool Render( const Scene& i_scene, Image& o_image ) override;

protected:
    // Allocate memory and data structures for rendering.
    virtual bool _Initialize() override;
};

TOY_NS_CLOSE
