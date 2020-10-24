//
// This file is auto-generated, please do not modify directly!
//

#pragma once

/// \file functions/viewportTransform.h
/// \ingroup gm_functions_linearAlgebra
///
/// Viewport transformation.
///
/// A viewport transformation transforms clip space coordinates into 2D viewport coordinates.
/// It is used to place a 2D image onto a viewport.

#include <gm/gm.h>

#include <gm/types/mat4f.h>
#include <gm/types/vec2f.h>

#include <gm/functions/matrixProduct.h>
#include <gm/functions/setScale.h>
#include <gm/functions/setTranslate.h>

GM_NS_OPEN

/// Construct an viewport transformation matrix.
/// \ingroup gm_functions_linearAlgebra
///
/// \param i_dimensions Dimensions (width, height) of the 2D image to show on the viewport.
/// \param i_offset Offset (X, Y) of the 2D image, in viewport space.
///
/// \return Viewport transformation matrix.
GM_HOST_DEVICE inline Mat4f ViewportTransform( const Vec2f& i_dimensions, const Vec2f& i_offset )
{
    Mat4f scaleXform = Mat4f::Identity();
    SetScale( Vec3f( i_dimensions.X() * 0.5f, i_dimensions.Y() * 0.5f, 1 ), scaleXform );

    Mat4f translateXform = Mat4f::Identity();
    SetTranslate( Vec3f( ( i_dimensions.X() * 0.5f ) + i_offset.X(), ( i_dimensions.Y() * 0.5f ) + i_offset.Y(), 0 ),
                  translateXform );

    return MatrixProduct( translateXform, scaleXform );
}

GM_NS_CLOSE