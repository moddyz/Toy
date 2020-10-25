#pragma once

/// \file rendering/flip.h
///
/// Flip an image across its vertical axis.

#include <tri/base/bitMask.h>
#include <tri/rendering/frameBuffer.h>

#include <gm/types/vec3f.h>

TRI_NS_OPEN

/// \enum FlipAxis
///
/// Axis to flip the image across.
enum FlipAxis : char
{
    FlipAxis_None = 0,      // binary 0000
    FlipAxis_X    = 1 << 0, // binary 0001
    FlipAxis_Y    = 1 << 1  // binary 0010
};

TRI_ENUM_BITMASK_OPERATORS( FlipAxis );

/// Flip or \em reflect an image across one or two axis.
///
/// \param i_axis One or more axis to flip.
/// \param i_image The input image.
/// \param o_image The output flipped image.
template < typename FrameBufferT >
void FlipImage( FlipAxis i_axis, const FrameBufferT& i_image, FrameBufferT& o_image )
{
    TRI_VERIFY( i_image.GetBuffer() != o_image.GetBuffer() );

    int xBegin, xEnd, xStep;
    if ( i_axis & FlipAxis_X )
    {
        xBegin = i_image.GetWidth() - 1;
        xEnd   = -1;
        xStep  = -1;
    }
    else
    {
        xBegin = 0;
        xEnd   = i_image.GetWidth();
        xStep  = 1;
    }

    int yBegin, yEnd, yStep;
    if ( i_axis & FlipAxis_Y )
    {
        yBegin = i_image.GetHeight() - 1;
        yEnd   = -1;
        yStep  = -1;
    }
    else
    {
        yBegin = 0;
        yEnd   = i_image.GetHeight();
        yStep  = 1;
    }

    for ( int z = 0; z < i_image.GetDepth(); z += 1 )
    {
        for ( int y = yBegin; y < yEnd; y += yStep )
        {
            for ( int x = xBegin; x < xEnd; x += xStep )
            {
                o_image( x, y, z) = i_image( x, y, z );
            }
        }
    }
}

TRI_NS_CLOSE
