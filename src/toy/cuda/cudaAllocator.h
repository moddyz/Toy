#pragma once

/// \file cuda/cudaAllocator.h
///
/// CUDA memory allocation.

#include <toy/toy.h>
#include <toy/core/diagnostic.h>
#include <toy/cuda/error.h>

TOY_NS_OPEN

/// \class CUDAAllocator
///
/// Memory allocation factilies for CUDA.
class CUDAAllocator
{
public:
    /// Allocate a block of managed CUDA memory.
    ///
    /// \param i_numBytes Number of bytes to allocate.
    ///
    /// \return The host pointer referring to the allocated memory on the device.
    static inline void* Allocate( size_t i_numBytes )
    {
        TOY_ASSERT( i_numBytes != 0 );
        void* devicePtr = nullptr;
        if ( cudaMallocManaged( &devicePtr, i_numBytes ) != cudaSuccess )
        {
            return nullptr;
        }

        return devicePtr;
    }
};

TOY_NS_CLOSE
