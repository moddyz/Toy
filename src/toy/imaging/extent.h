#pragma once

/// \file imaging/convert.h
///
/// Utilities for converting between in-memory image data formats.

#include <toy/memory/matrix.h>

#include <gm/types/vec2iRange.h>

TOY_NS_OPEN

/// Compute the extent of an image.
///
/// \param i_image The image.
///
/// \return The extent or bounds of the image.
template< typename MatrixT >
gm::Vec2iRange GetImageExtent( const MatrixT& i_image )
{
    return gm::Vec2iRange( gm::Vec2i( 0, 0 ), gm::Vec2i( i_image.GetColumns(), i_image.GetRows() ) );
}

TOY_NS_CLOSE


