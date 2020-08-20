#pragma once

/// \file core/cudaAllocator.h
///
/// CUDA memory allocation.

#include <toy/toy.h>
#include <toy/core/diagnostic.h>
#include <toy/cuda/diagnostic.h>

TOY_NS_OPEN

/// \class CUDAAllocator
///
/// Memory allocation factilies for CUDA.
class CUDAAllocator : Allocator
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
        cuda cudaMallocManaged( &devicePtr, i_numBytes,
    }
};

TOY_NS_CLOSE
