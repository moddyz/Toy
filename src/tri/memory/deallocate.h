#pragma once

/// \file memory/deallocate.h
///
/// Memory de-allocation.

#include <tri/base/diagnostic.h>
#include <tri/memory/cudaError.h>
#include <tri/memory/residency.h>

#include <cstring>
#include <cuda_runtime.h>

TRI_NS_OPEN

/// \struct MemoryDeallocate
///
/// Template prototype for a memory allocation operation.
template < Residency ResidencyT >
struct MemoryDeallocate
{
};

template <>
struct MemoryDeallocate< Host >
{
    static inline bool Execute( void* o_buffer )
    {
        TRI_ASSERT( o_buffer != nullptr );
        free( o_buffer );
        return true;
    }
};

template <>
struct MemoryDeallocate< CUDA >
{
    static inline bool Execute( void* o_buffer )
    {
        TRI_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK( cudaFree( o_buffer ) );
    }
};

TRI_NS_CLOSE
