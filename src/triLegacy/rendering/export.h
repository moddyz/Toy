#pragma once

/// \file rendering/export.h
///
/// Utilities for exporting an in-memory image onto disk.

#include <tri/rendering/frameBuffer.h>

#include <gm/types/vec3f.h>

TRI_NS_OPEN

/// Write an JPEG image onto disk.
///
/// \param i_imageBuffer Image buffer with normalized [0,1) RGB floating point values.
/// \param i_filePath The file path to write the image to.
///
/// \return Whether the image was successfully written to disk.
bool ExportJpeg( const FrameBuffer< gm::Vec3f, Host >& i_frameBuffer, const std::string& i_filePath );

TRI_NS_CLOSE
