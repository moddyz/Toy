#pragma once

/// \file memory/copy.h
///
/// Memory copy operation.
///
/// Definition of the copy operations between the various memory residencies.
/// These operations are all synchronous.  A more specific API is required for
/// taking advantage of asynchronous CUDA copies.

#include <tri/base/diagnostic.h>
#include <tri/memory/residency.h>

TRI_NS_OPEN

/// \struct MemoryCopy
///
/// Template prototype for a copy operation.
template < Residency SrcResidencyT, Residency DstResidencyT >
struct MemoryCopy
{
};

/// Specialization for host -> host copy.
template <>
struct MemoryCopy< Host, Host >
{
    static inline bool Execute( size_t i_numBytes, const void* i_srcBuffer, void* o_dstBuffer )
    {
        TRI_ASSERT( o_buffer != nullptr );
        return memcpy( o_dstBuffer, i_srcBuffer, i_numBytes ) != nullptr;
    }
};

/// Specialization for synchronous cuda -> cuda copy.
template <>
struct MemoryCopy< CUDA, CUDA >
{
    static inline bool Execute( size_t i_numBytes, const void* i_srcBuffer, void* o_dstBuffer )
    {
        TRI_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK( cudaMemcpy( o_dstBuffer, i_srcBuffer, i_numBytes, cudaMemcpyDeviceToDevice ) );
    }
};

/// Specialization for synchronous host -> cuda copy.
template <>
struct MemoryCopy< Host, CUDA >
{
    static inline bool Execute( size_t i_numBytes, const void* i_srcBuffer, void* o_dstBuffer )
    {
        TRI_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK( cudaMemcpy( o_dstBuffer, i_srcBuffer, i_numBytes, cudaMemcpyHostToDevice ) );
    }
};

/// Specialization for synchronous cuda -> host copy.
template <>
struct MemoryCopy< CUDA, Host >
{
    static inline bool Execute( size_t i_numBytes, const void* i_srcBuffer, void* o_dstBuffer )
    {
        TRI_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK( cudaMemcpy( o_dstBuffer, i_srcBuffer, i_numBytes, cudaMemcpyDeviceToHost ) );
    }
};

TRI_NS_CLOSE
