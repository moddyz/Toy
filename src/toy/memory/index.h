#pragma once

/// \file memory/index.h
///
/// Memory indexed access operator.

#include <toy/memory/residency.h>
#include <toy/base/diagnostic.h>
#include <toy/base/falseType.h>

#include <cuda_runtime.h>

TOY_NS_OPEN

/// \struct MemoryIndex
///
/// Template prototype for a buffer index operation.
template < Residency ResidencyT >
struct MemoryIndex
{
};

template <>
struct MemoryIndex< Host >
{
    template < typename ValueT >
    static inline ValueT& Execute( ValueT* i_buffer, size_t i_index )
    {
        return i_buffer[ i_index ];
    }
};

template <>
struct MemoryIndex< CUDA >
{
    template < typename ValueT >
    static inline ValueT& Execute( ValueT* i_buffer, size_t i_index )
    {
        static_assert( FalseType< ValueT >::value, "Cannot index into cuda buffer from host code." );
        return ValueT();
    }
};

TOY_NS_CLOSE
