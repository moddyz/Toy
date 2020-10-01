#pragma once

/// \file imaging/transformPoints.h

#include <toy/memory/array.h>

#include <gm/functions/transformPoint.h>
#include <gm/types/mat4f.h>
#include <gm/types/vec3f.h>

#include <array>

#include <cuda_runtime.h>

TOY_NS_OPEN

/// \struct TransformPoints
///
/// Template prototype for a linear transformation performed on a points.
template < Residency ResidencyT >
struct TransformPoints
{
};

template <>
struct TransformPoints< Host >
{
    static inline bool Execute( const gm::Mat4f&                i_transform,
                                const Array< gm::Vec3f, Host >& i_points,
                                Array< gm::Vec3f, Host >&       o_points )
    {
        TOY_ASSERT( i_points.size() == o_points.size() );

        for ( size_t pointIndex = 0; pointIndex < i_points.GetSize(); pointIndex++ )
        {
            o_points[ pointIndex ] = gm::TransformPoint( i_transform, i_points[ pointIndex ] );
        }

        return true;
    }
};

// Function prototype for the associated cuda kernel.
__global__ void TransformPoints_Kernel( size_t           i_numPoints,
                                        const gm::Mat4f& i_transform,
                                        const gm::Vec3f* i_points,
                                        gm::Vec3f*       o_points );

template <>
struct TransformPoints< Cuda >
{
    static inline bool Execute( const gm::Mat4f&                i_transform,
                                const Array< gm::Vec3f, Cuda >& i_points,
                                Array< gm::Vec3f, Cuda >&       o_points )
    {
        dim3 block( 256, 1, 1 );
        dim3 grid( ( i_points.GetSize() + block.x - 1 ) / block.x, 1, 1 );

        // Arguments
        size_t                 numPoints    = i_points.GetSize();
        gm::Vec3f*             inputPoints  = i_points.GetBuffer();
        gm::Vec3f*             outputPoints = o_points.GetBuffer();
        std::array< void*, 4 > args         = {&numPoints,
                                       const_cast< gm::Mat4f* >( &i_transform ),
                                       &inputPoints,
                                       &outputPoints};

        return cudaLaunchKernel( ( void* ) TransformPoints_Kernel, grid, block, args.data(), 0, nullptr ) ==
               cudaSuccess;
    }
};

TOY_NS_CLOSE
