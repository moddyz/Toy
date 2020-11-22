#include "bufferUtils.h"
#include "objectContainer.h"

#include <cassert>

/// \var s_buffers
///
/// Global internal container of Tri_Buffer objects.
///
/// TODO: Exert explicit control over lifetime of this container.
static Tri_ObjectContainer<Tri_Buffer> s_buffers;

Tri_Buffer*
Tri_BufferCreate(TriBuffer& buffer,
                 const Tri_Context* context,
                 const gm::Vec3i& dims,
                 TriFormat format,
                 TriDevice device,
                 void* bufferPtr)
{
    typename decltype(s_buffers)::EntryT entry = s_buffers.Create<Tri_Buffer>();

    // Populate internal buffer info.
    Tri_Buffer* internalBuffer = entry.second;
    internalBuffer->dims = dims;
    internalBuffer->format = format;
    internalBuffer->device = device;
    internalBuffer->ptr = bufferPtr;
    internalBuffer->context = context;

    // Populate API buffer info.
    buffer.id = entry.first;

    return entry.second;
}

Tri_Buffer*
Tri_BufferGet(TriId id)
{
    return s_buffers.Get(id);
}

bool
Tri_BufferDelete(TriBuffer& buffer)
{
    if (s_buffers.Delete(buffer.id)) {
        buffer = TriBuffer();
        return TriStatus_Success;
    } else {
        return TriStatus_ContextNotFound;
    }
}

size_t
Tri_FormatGetNumBytes(TriFormat format)
{
    switch (format) {
        case TriFormat_Float32:
            return sizeof(float);
        case TriFormat_Float32_Vec2:
            return sizeof(float) * 2;
        case TriFormat_Float32_Vec3:
            return sizeof(float) * 3;
        case TriFormat_Float32_Vec4:
            return sizeof(float) * 4;
        case TriFormat_Uint32:
            return sizeof(uint32_t);
        case TriFormat_Uint8_Vec3:
            return sizeof(uint8_t) * 3;
        case TriFormat_Uint8_Vec4:
            return sizeof(uint8_t) * 4;
        case TriFormat_Uninitialized:
        default:
            assert(false);
            return 0;
    }
}

size_t
Tri_BufferComputeNumBytes(const gm::Vec3i& dims, TriFormat format)
{
    return dims.X() * dims.Y() * dims.Z() * Tri_FormatGetNumBytes(format);
}
