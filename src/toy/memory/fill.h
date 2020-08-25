#pragma once

/// \file memory/fill.h
///
/// Memory fill operation.

#include <toy/memory/allocate.h>
#include <toy/memory/copy.h>
#include <toy/memory/residency.h>
#include <toy/utils/diagnostic.h>

#include <cuda_runtime.h>

TOY_NS_OPEN

/// \struct Fill
///
/// Template prototype for a fill operation.
template < Residency ResidencyT >
struct Fill
{
};

template <>
struct Fill< Host >
{
    template < typename ValueT >
    static inline bool Execute( size_t i_numElements, const ValueT& i_value, ValueT* o_dstBuffer )
    {
        TOY_ASSERT( o_dstBuffer != nullptr );

        for ( size_t elementIndex = 0; elementIndex < i_numElements; ++elementIndex )
        {
            o_dstBuffer[ elementIndex ] = i_value;
        }

        return true;
    }
};

template <>
struct Fill< Cuda >
{
    template < typename ValueT >
    static inline bool Execute( size_t i_numElements, const ValueT& i_value, ValueT* o_dstBuffer )
    {
        // Allocate and fill on the host.
        size_t numBytes  = i_numElements * sizeof( ValueT );
        void*  srcBuffer = Allocate< Host >::Execute( numBytes );
        if ( srcBuffer == nullptr )
        {
            return false;
        }

        TOY_VERIFY( Fill< Host >::Execute( i_numElements, i_value, static_cast< ValueT* >( srcBuffer ) ) );

        // Then copy to GPU.
        bool result = Copy< Host, Cuda >::Execute( numBytes, srcBuffer, o_dstBuffer );

        // Delete temporary buffer.
        TOY_VERIFY( Deallocate< Host >::Execute( srcBuffer ) );

        return result;
    }
};

TOY_NS_CLOSE
