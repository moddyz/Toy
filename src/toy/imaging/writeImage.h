#pragma once

/// \file imaging/writer.h
///
/// Utilities for writing a image to disk.

#include <toy/memory/matrix.h>

#include <gm/types/vec3f.h>

TOY_NS_OPEN

/// Write an RGB image to disk.
///
/// \param i_imageBuffer Image buffer with normalized [0,1) RGB floating point values.
/// \param i_filePath The file path to write the image to.
///
/// \return Whether the image was successfully written to disk.
bool WriteRGBImage( const Matrix< gm::Vec3f, Host >& i_imageBuffer, const std::string& i_filePath );

TOY_NS_CLOSE

