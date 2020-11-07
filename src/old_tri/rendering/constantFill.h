#pragma once

/// \file rendering/constantFill.h
///
/// Utilities for filling buffers with a constant value.

#include <tri/rendering/frameBuffer.h>

#include <gm/types/vec3f.h>

#include <array>

TRI_NS_OPEN

/// \struct ConstantFill
///
/// Template prototype for a
template < Residency ResidencyT >
struct ConstantFill
{
};

template <>
struct ConstantFill< Host >
{
    template < typename ValueT >
    static inline bool Execute( size_t i_numElements, const ValueT& i_value, ValueT* o_buffer )
    {
        for ( size_t index = 0; index < i_numElements; ++index )
        {
            o_buffer[ index ] = i_value;
        }

        return true;
    }
};

// Function prototypes for the associated cuda kernels.
template < typename ValueT >
__global__ void ConstantFill_Kernel( size_t i_numElements, const ValueT i_value, ValueT* o_buffer );

template <>
struct ConstantFill< CUDA >
{
    template < typename ValueT >
    static inline bool Execute( size_t i_numElements, const ValueT& i_value, ValueT* o_buffer )
    {
        dim3 block( 256, 1, 1 );
        dim3 grid( ( i_numElements + block.x - 1 ) / block.x, 1, 1 );

        // Arguments
        std::array< void*, 3 > args = {&i_numElements, const_cast< ValueT* >( &i_value ), &o_buffer};
        return cudaLaunchKernel( ( void* ) ConstantFill_Kernel< ValueT >, grid, block, args.data(), 0, nullptr ) ==
               cudaSuccess;
    }
};

TRI_NS_CLOSE
// Clear.

