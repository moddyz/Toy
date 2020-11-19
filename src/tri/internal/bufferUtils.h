#pragma once

#include <tri/tri.h>

/// Get the number of bytes for value of \p format.
///
/// \param format The data format.
size_t
Tri_FormatGetNumBytes(TriFormat format);

/// Compute the number of bytes required to allocate a buffer with \p numElements.
///
/// \param size Number of elements in the buffer.
/// \param format The format / data type layout of each pixel.
size_t
Tri_BufferComputeNumBytes(int numElements, TriFormat format);

/// Compute the number of bytes required to allocate a buffer with \p numElements.
///
/// \param width Pixel width.
/// \param height Pixel height.
size_t
Tri_Buffer2DComputeNumBytes(int width, int height, TriFormat format);
