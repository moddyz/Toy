#pragma once

/// \file memory/deallocate.h
///
/// Memory de-allocation.

#include <toy/memory/cudaError.h>
#include <toy/memory/residency.h>
#include <toy/utils/diagnostic.h>

#include <cstring>
#include <cuda_runtime.h>

TOY_NS_OPEN

/// \struct Deallocate
///
/// Template prototype for a memory allocation operation.
template < Residency ResidencyT >
struct Deallocate
{
};

template <>
struct Deallocate< Host >
{
    static inline bool Execute( void* o_buffer )
    {
        TOY_ASSERT( o_buffer != nullptr );
        free( o_buffer );
        return true;
    }
};

template <>
struct Deallocate< Cuda >
{
    static inline bool Execute( void* o_buffer )
    {
        TOY_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK( cudaFree( o_buffer ) );
    }
};

TOY_NS_CLOSE
