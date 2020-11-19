#include "bufferUtils.h"

#include <cassert>

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
Tri_BufferComputeNumBytes(size_t numElements, TriFormat format)
{
    return numElements * Tri_FormatGetNumBytes(format);
}

size_t
Tri_Buffer2DComputeNumBytes(int width, int height, TriFormat format)
{
    return Tri_BufferComputeNumBytes(width * height, format);
}
