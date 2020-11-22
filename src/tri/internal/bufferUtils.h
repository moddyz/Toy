#pragma once

#include <tri/tri.h>

#include <gm/types/vec3i.h>

// Forward declaration.
class Tri_Context;

/// \class Tri_Buffer
///
/// Internal buffer representation.
struct Tri_Buffer
{
    /// Dimensions of the buffer elements.
    gm::Vec3i dims;

    /// The data layout of each element in the buffer.
    TriFormat format{ TriFormat_Uninitialized };

    /// The current device where the buffer memory resides.
    ///
    /// This does \em not have match the device of the context, as
    /// user-mapped buffers can reside on a different device.
    TriDevice device{ TriDevice_Uninitialized };

    /// Pointer to the starting address of the buffer.
    void* ptr{ nullptr };

    /// Associated context.
    const Tri_Context* context{ nullptr };
};

/// Create an internal buffer representation, and register client TriBuffer id.
Tri_Buffer*
Tri_BufferCreate(TriBuffer& buffer,
                 const Tri_Context* context,
                 const gm::Vec3i& dims,
                 TriFormat format,
                 TriDevice device,
                 void* bufferPtr);

/// Query an internal buffer represenation.
Tri_Buffer*
Tri_BufferGet(TriId id);

/// Delete the internal buffer representation.
bool
Tri_BufferDelete(TriBuffer& buffer);

/// Get the number of bytes for value of \p format.
///
/// \param format The data format.
size_t
Tri_FormatGetNumBytes(TriFormat format);

/// Compute the number of bytes required to allocate a buffer with \p
/// numElements.
///
/// \param size Number of elements in the buffer.
/// \param format The format / data type layout of each pixel.
size_t
Tri_BufferComputeNumBytes(const gm::Vec3i& dims, TriFormat format);
