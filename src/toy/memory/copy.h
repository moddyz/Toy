#pragma once

/// \file memory/copy.h
///
/// Memory copy operation.
///
/// Definition of the copy operations between the various memory residencies.
/// These operations are all synchronous.  A more specific API is required for
/// taking advantage of asynchronous CUDA copies.

#include <toy/memory/residency.h>
#include <toy/utils/diagnostic.h>

TOY_NS_OPEN

/// \struct Copy
///
/// Template prototype for a copy operation.
template < Residency SrcResidencyT, Residency DstResidencyT >
struct Copy
{
};

/// Specialization for host -> host copy.
template <>
struct Copy< Host, Host >
{
    static inline bool Execute( size_t i_numBytes, const void* i_srcBuffer, void* o_dstBuffer )
    {
        TOY_ASSERT( o_buffer != nullptr );
        return memcpy( o_dstBuffer, i_srcBuffer, i_numBytes ) != nullptr;
    }
};

/// Specialization for synchronous cuda -> cuda copy.
template <>
struct Copy< Cuda, Cuda >
{
    static inline bool Execute( size_t i_numBytes, const void* i_srcBuffer, void* o_dstBuffer )
    {
        TOY_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK( cudaMemcpy( o_dstBuffer, i_srcBuffer, i_numBytes, cudaMemcpyDeviceToDevice ) );
    }
};

/// Specialization for synchronous host -> cuda copy.
template <>
struct Copy< Host, Cuda >
{
    static inline bool Execute( size_t i_numBytes, const void* i_srcBuffer, void* o_dstBuffer )
    {
        TOY_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK( cudaMemcpy( o_dstBuffer, i_srcBuffer, i_numBytes, cudaMemcpyHostToDevice ) );
    }
};

/// Specialization for synchronous cuda -> host copy.
template <>
struct Copy< Cuda, Host >
{
    static inline bool Execute( size_t i_numBytes, const void* i_srcBuffer, void* o_dstBuffer )
    {
        TOY_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK( cudaMemcpy( o_dstBuffer, i_srcBuffer, i_numBytes, cudaMemcpyDeviceToHost ) );
    }
};

TOY_NS_CLOSE
