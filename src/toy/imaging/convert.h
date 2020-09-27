#pragma once

/// \file imaging/convert.h
///
/// Utilities for converting between in-memory image data formats.

#include <toy/memory/matrix.h>

#include <gm/types/vec3f.h>

TOY_NS_OPEN

/// Convert an RGB floating point image into RGBA uint32_t.
///
/// \param i_image The input image, with RGB floating point encoding.
/// \param o_image The output image, with RGBA uint32_t encoding.
///
/// \return Whether the image conversion was successful;
void ConvertImageVec3fToUint32( const Matrix< gm::Vec3f, Host >& i_image, Matrix< uint32_t, Host >& o_image );

TOY_NS_CLOSE
