#pragma once

/// \file memory/allocate.h
///
/// Memory allocation.

#include <toy/memory/cudaError.h>
#include <toy/memory/residency.h>
#include <toy/utils/diagnostic.h>

#include <cstring>
#include <cuda_runtime.h>

TOY_NS_OPEN

/// \struct Allocate
///
/// Template prototype for a memory allocation operation.
template < Residency ResidencyT >
struct Allocate
{
};

template <>
struct Allocate< Host >
{
    static inline void* Execute( size_t i_numBytes )
    {
        TOY_ASSERT( i_numBytes != 0 );
        return malloc( i_numBytes );
    }
};

template <>
struct Allocate< Cuda >
{
    static inline void* Execute( size_t i_numBytes )
    {
        TOY_ASSERT( i_numBytes != 0 );
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

TOY_NS_CLOSE
