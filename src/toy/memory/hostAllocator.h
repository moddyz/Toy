#pragma once

/// \file memory/hostAllocator.h
///
/// Host memory allocation.

#include <toy/utils/diagnostic.h>

#include <cstring>

TOY_NS_OPEN

/// \class HostAllocator
///
/// Memory allocation factilies for Host.
class HostAllocator
{
public:
    /// Allocate a block of host memory.
    ///
    /// \return The host pointer referring to the allocated RAM.
    static inline void* Allocate( size_t i_numBytes )
    {
        TOY_ASSERT( i_numBytes != 0 );
        return malloc( i_numBytes );
    }

    /// De-allocate a block of host memory.
    ///
    /// \param o_buffer Pointer to the memory to de-allocate.
    ///
    /// \return Whether the deallocation was performed successfully.
    static inline bool Deallocate( void* o_buffer )
    {
        TOY_ASSERT( o_buffer != nullptr );
        free( o_buffer );
        return true;
    }
};

TOY_NS_CLOSE
