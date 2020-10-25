#pragma once

/// \file rendering/convert.h
///
/// Utilities for converting between in-memory image data formats.

#include <tri/rendering/frameBuffer.h>

#include <gm/types/vec2iRange.h>

TRI_NS_OPEN

/// Compute the extent of an image.
///
/// \param i_image The image.
///
/// \return The extent or bounds of the image.
template < typename FrameBufferT >
gm::Vec2iRange GetImageExtent( const FrameBufferT& i_image )
{
    return gm::Vec2iRange( gm::Vec2i( 0, 0 ), gm::Vec2i( i_image.GetColumns(), i_image.GetRows() ) );
}

TRI_NS_CLOSE
