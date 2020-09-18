#pragma once

#include <toy/toy.h>

#include <gm/types/vec2i.h>

TOY_NS_OPEN

/// \class RenderSettings
///
/// A collection of settings for configuring the renderer.

class RenderSettings
{
public:
    gm::Vec2i m_imageDimensions;
};

TOY_NS_CLOSE
