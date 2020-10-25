#pragma once

/// \file rendering/flip.h
///
/// Flip an image across its vertical axis.

#include <tri/base/bitMask.h>
#include <tri/memory/matrix.h>

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
template < typename MatrixT >
void FlipImage( FlipAxis i_axis, const MatrixT& i_image, MatrixT& o_image )
{
    TRI_VERIFY( i_image.GetBuffer() != o_image.GetBuffer() );
    o_image.Resize( i_image.GetRows(), i_image.GetColumns() );

    int xBegin, xEnd, xStep;
    if ( i_axis & FlipAxis_X )
    {
        xBegin = i_image.GetColumns() - 1;
        xEnd   = -1;
        xStep  = -1;
    }
    else
    {
        xBegin = 0;
        xEnd   = i_image.GetColumns();
        xStep  = 1;
    }

    int yBegin, yEnd, yStep;
    if ( i_axis & FlipAxis_Y )
    {
        yBegin = i_image.GetRows() - 1;
        yEnd   = -1;
        yStep  = -1;
    }
    else
    {
        yBegin = 0;
        yEnd   = i_image.GetRows();
        yStep  = 1;
    }

    for ( int yCoord = yBegin; yCoord < yEnd; yCoord += yStep )
    {
        for ( int xCoord = xBegin; xCoord < xEnd; xCoord += xStep )
        {
            o_image( yCoord, xCoord ) = i_image( yCoord, xCoord );
        }
    }
}

TRI_NS_CLOSE
