#pragma once

/// \file rendering/formatConversion.h
///
/// Utilities for converting between image storage formats.

#include <tri/rendering/frameBuffer.h>

#include <gm/types/vec3f.h>

#include <array>

TRI_NS_OPEN

/// \struct ConvertRGBFloatToRGBAUint32
///
/// Template prototype for a linear transformation performed on a points.
template < Residency ResidencyT >
struct ConvertRGBFloatToRGBAUint32
{
};

template <>
struct ConvertRGBFloatToRGBAUint32< Host >
{
    static inline bool Execute( size_t i_numPixels, const gm::Vec3f* i_image, uint32_t* o_image )
    {
        for ( size_t pixelIndex = 0; pixelIndex < i_numPixels; ++pixelIndex )
        {
            const gm::Vec3f& inPixel  = i_image[ pixelIndex ];
            uint8_t*         outPixel = reinterpret_cast< uint8_t* >( &o_image[ pixelIndex ] );
            outPixel[ 0 ]             = static_cast< uint8_t >( 255.999 * inPixel[ 0 ] );
            outPixel[ 1 ]             = static_cast< uint8_t >( 255.999 * inPixel[ 1 ] );
            outPixel[ 2 ]             = static_cast< uint8_t >( 255.999 * inPixel[ 2 ] );
        }

        return true;
    }
};

// Function prototype for the associated cuda kernel.
__global__ void ConvertRGBFloatToRGBAUint32_Kernel( size_t i_numPixels, const gm::Vec3f* i_image, uint32_t* o_points );

template <>
struct ConvertRGBFloatToRGBAUint32< CUDA >
{
    static inline bool Execute( size_t i_numPixels, const gm::Vec3f* i_image, uint32_t* o_image )
    {
        dim3 block( 256, 1, 1 );
        dim3 grid( ( i_numPixels + block.x - 1 ) / block.x, 1, 1 );

        // Arguments
        std::array< void*, 3 > args = {&i_numPixels, &i_image, &o_image};
        return cudaLaunchKernel( ( void* ) ConvertRGBFloatToRGBAUint32_Kernel, grid, block, args.data(), 0, nullptr ) ==
               cudaSuccess;
    }
};

TRI_NS_CLOSE
