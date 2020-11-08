#pragma once

/// \file memory/allocate.h
///
/// Memory allocation.

#include <tri/base/diagnostic.h>
#include <tri/memory/cudaError.h>
#include <tri/memory/residency.h>

#include <cstring>
#include <cuda_runtime.h>

TRI_NS_OPEN

/// \struct MemoryAllocate
///
/// Template prototype for a memory allocation operation.
template < Residency ResidencyT >
struct MemoryAllocate
{
};

template <>
struct MemoryAllocate< Host >
{
    static inline void* Execute( size_t i_numBytes )
    {
        TRI_ASSERT( i_numBytes != 0 );
        return malloc( i_numBytes );
    }
};

template <>
struct MemoryAllocate< CUDA >
{
    static inline void* Execute( size_t i_numBytes )
    {
        TRI_ASSERT( i_numBytes != 0 );
        void* devicePtr = nullptr;
        if ( CUDA_CHECK( cudaMallocManaged( &devicePtr, i_numBytes ) ) )
        {
            return devicePtr;
        }
        else
        {
            return nullptr;
        }
    }
};

TRI_NS_CLOSE
