#pragma once

#include <tri/base/diagnostic.h>
#include <tri/memory/allocate.h>
#include <tri/memory/copy.h>
#include <tri/memory/residency.h>

#include <cuda_runtime.h>

TRI_NS_OPEN

/// \struct MemoryCompare
///
/// Template prototype for a memory comparison operation between two arrays.
template < Residency ResidencyT >
struct MemoryCompare
{
};

template <>
struct MemoryCompare< Host >
{
    template < typename ValueT >
    static inline bool Execute( size_t i_numElements, const ValueT* i_arrayA, const ValueT* i_arrayB )
    {
        TRI_ASSERT( i_arrayA != nullptr );
        TRI_ASSERT( i_arrayB != nullptr );

        for ( size_t index = 0; index < i_numElements; ++index )
        {
            if ( i_arrayA[ index ] != i_arrayB[ index ] )
            {
                return false;
            }
        }

        return true;
    }
};

template <>
struct MemoryCompare< CUDA >
{
    template < typename ValueT >
    static inline bool Execute( size_t i_numElements, const ValueT* i_arrayA, const ValueT* i_arrayB )
    {
        // MemoryAllocate host buffers.
        size_t  numBytes   = i_numElements * sizeof( ValueT );
        ValueT* hostArrayA = ( ValueT* ) MemoryAllocate< Host >::Execute( numBytes );
        if ( hostArrayA == nullptr )
        {
            return false;
        }
        ValueT* hostArrayB = ( ValueT* ) MemoryAllocate< Host >::Execute( numBytes );
        if ( hostArrayB == nullptr )
        {
            return false;
        }

        // Copy to host buffers.
        bool result = MemoryCopy< CUDA, Host >::Execute( numBytes, i_arrayA, hostArrayA );
        TRI_VERIFY( result );
        result = MemoryCopy< CUDA, Host >::Execute( numBytes, i_arrayB, hostArrayB );
        TRI_VERIFY( result );

        // Do comparison on Host.
        return MemoryCompare< Host >::Execute( i_numElements, hostArrayA, hostArrayB );
    }
};

TRI_NS_CLOSE
