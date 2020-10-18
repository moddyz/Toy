//
// This file is auto-generated, please do not modify directly!
//

#pragma once

/// \file functions/orthographicProjection.h
/// \ingroup gm_functions_linearAlgebra
///
/// Orthographic projection.
///
/// Construct an transformation matrix which projects vertices into a viewing volume.

#include <gm/gm.h>

#include <gm/types/mat4f.h>
#include <gm/types/vec3fRange.h>

#include <gm/functions/matrixProduct.h>
#include <gm/functions/setScale.h>
#include <gm/functions/setTranslate.h>

GM_NS_OPEN

/// Construct an orthographic projection matrix from an axis-aligned, rectilinear viewing volume.
/// \ingroup gm_functions_linearAlgebra
///
/// \param i_viewingVolume Orthographic viewing volume.
///
/// \return Orthographic projection transformation matrix.
GM_HOST_DEVICE inline Mat4f OrthographicProjection( const Vec3fRange& i_viewingVolume )
{
    // Center viewing volume about origin, such that the scaling is applied uniformly.
    gm::Mat4f centeringXform = gm::Mat4f::Identity();
    SetTranslate( gm::Vec3f( -( i_viewingVolume.Max().X() + i_viewingVolume.Min().X() ) * 0.5f,
                             -( i_viewingVolume.Max().Y() + i_viewingVolume.Min().Y() ) * 0.5f,
                             -( i_viewingVolume.Max().Z() + i_viewingVolume.Min().Z() ) * 0.5f ),
                  centeringXform );

    // Scale viewing volume into a volume of min=(-1, -1, -1), max=(1, 1, 1)
    gm::Mat4f scaleXform = gm::Mat4f::Identity();
    SetScale( gm::Vec3f( 2.0f / ( i_viewingVolume.Max().X() - i_viewingVolume.Min().X() ),
                         2.0f / ( i_viewingVolume.Max().Y() - i_viewingVolume.Min().Y() ),
                         2.0f / ( i_viewingVolume.Max().Z() - i_viewingVolume.Min().Z() ) ),
              scaleXform );

    return gm::MatrixProduct( scaleXform, centeringXform );
}

GM_NS_CLOSE