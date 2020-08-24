#pragma once

/// \file memory/cudaAllocator.h
///
/// CUDA memory allocation.

#include <toy/utils/diagnostic.h>

#include <toy/memory/cudaError.h>

TOY_NS_OPEN

/// \class CudaAllocator
///
/// Memory allocation factilies for CUDA.
class CudaAllocator
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
        if ( CUDA_CHECK_ERROR( cudaMallocManaged( &devicePtr, i_numBytes ) ) )
        {
            return devicePtr;
        }
        else
        {
            return nullptr;
        }
    }

    /// De-allocate a block of host memory.
    ///
    /// \param o_buffer Pointer to the memory to de-allocate.
    ///
    /// \return Whether the copy was performed successfully.
    static inline bool Deallocate( void* o_buffer )
    {
        TOY_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK_ERROR( cudaFree( o_buffer ) );
    }
};

TOY_NS_CLOSE
