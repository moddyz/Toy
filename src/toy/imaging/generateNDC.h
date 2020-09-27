#pragma once

/// \file imaging/generateNDC.h

#include <toy/memory/matrix.h>

#include <gm/types/vec2f.h>

#include <array>

#include <cuda_runtime.h>

TOY_NS_OPEN

/// \struct GenerateNDC
///
/// Template prototype for a operation which produces normalized device coordiantes
/// for each ray sample.
template < Residency ResidencyT >
struct GenerateNDC
{
};

template <>
struct GenerateNDC< Host >
{
    static inline bool Execute( const gm::Vec2iRange& i_extent, Matrix< gm::Vec2f, Host >& o_pixelCoords )
    {
        gm::Vec2i size = i_extent.Max() - i_extent.Min();
        o_pixelCoords.Resize( size.Y(), size.X() );

        float rowsInverse = 1.0f / o_pixelCoords.GetRows();
        float colsInverse = 1.0f / o_pixelCoords.GetColumns();

        for ( gm::Vec2i coord : i_extent )
        {
            o_pixelCoords( coord.Y(), coord.X() ) = gm::Vec2f( coord.X() * colsInverse, coord.Y() * rowsInverse );
        }

        return true;
    }
};

// Function prototype for the associated cuda kernel.
__global__ void GenerateNDC_Kernel( gm::Vec2f* o_coordinates );

template <>
struct GenerateNDC< Cuda >
{
    static inline bool Execute( const gm::Vec2iRange& i_extent, Matrix< gm::Vec2f, Cuda >& o_pixelCoords )
    {
        gm::Vec2i size = i_extent.Max() - i_extent.Min();
        o_pixelCoords.Resize( size.Y(), size.X() );

        dim3 block( 256, 256, 1 );
        dim3 grid;
        grid.x = ( o_pixelCoords.GetColumns() + block.x - 1 ) / block.x;
        grid.y = ( o_pixelCoords.GetRows() + block.y - 1 ) / block.y;

        // Arguments
        int                    rows        = o_pixelCoords.GetRows();
        int                    cols        = o_pixelCoords.GetColumns();
        float                  rowsInverse = 1.0f / o_pixelCoords.GetRows();
        float                  colsInverse = 1.0f / o_pixelCoords.GetColumns();
        gm::Vec2f*             buffer      = o_pixelCoords.GetBuffer();
        std::array< void*, 5 > args        = {&rows, &cols, &rowsInverse, &colsInverse, &buffer};

        return cudaLaunchKernel( ( void* ) GenerateNDC_Kernel, grid, block, args.data(), 0, nullptr ) == cudaSuccess;
    }
};

TOY_NS_CLOSE
