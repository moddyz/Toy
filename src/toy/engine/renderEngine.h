#pragma once

#include <toy/engine/renderSettings.h>
#include <toy/memory/matrix.h>
#include <toy/memory/residency.h>

TOY_NS_OPEN

/// \class RenderEngine
///
/// Base class for an integration of various models and operators to render an image.
template < Residency ResidencyT >
class RenderEngine
{
public:
    //-------------------------------------------------------------------------
    /// \name Construction
    //-------------------------------------------------------------------------

    /// Initialize a RenderEngine with default settings.
    RenderEngine();

    /// Initialize a RenderEngine with specified settings \p i_renderSettings.
    RenderEngine( const RenderSettings& i_renderSettings );

    //-------------------------------------------------------------------------
    /// \name Rendering
    //-------------------------------------------------------------------------

    /// \typedef Image
    ///
    /// Convenience type definition for a image data structure.
    using Image = Matrix< gm::Vec3f, ResidencyT >;

    /// Render an \p o_image based on \p i_scene.
    bool Render( const Scene& i_scene, Image& o_image ) = 0;

protected:
    /// Derived class should implement this to initialize this engine based on RenderSettings.
    /// Non-scene-specific memory allocations can be done here.
    virtual bool _Initialize() = 0;

private:
    RenderSettings m_renderSettings;
};

TOY_NS_CLOSE
