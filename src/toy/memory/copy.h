#pragma once

/// \file memory/residency.h
///
/// Operations for different combinations of source and destination memory residency.

#include <toy/memory/residency.h>
#include <toy/utils/diagnostic.h>

TOY_NS_OPEN

/// \struct CrossResidency
///
/// Template prototype for a copy operation.
template < Residency DstResidencyT, Residency SrcResidencyT >
struct Copy
{
};

/// Specialization for host -> host copy.
template <>
struct Copy< Host, Host >
{
    static inline bool Execute( void* i_dstBuffer, void* i_srcBuffer, size_t i_numBytes )
    {
        TOY_ASSERT( o_buffer != nullptr );
        return memcpy( i_dstBuffer, i_srcBuffer, i_numBytes ) != nullptr;
    }
};

/// Specialization for synchronous cuda -> cuda copy.
template <>
struct Copy< Cuda, Cuda >
{
    static inline bool Execute( void* i_dstBuffer, void* i_srcBuffer, size_t i_numBytes )
    {
        TOY_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK_ERROR( cudaMemcpy( i_dstBuffer, i_srcBuffer, i_numBytes, cudaMemcpyDeviceToDevice ) );
    }
};

/// Specialization for synchronous host -> cuda copy.
template <>
struct Copy< Host, Cuda >
{
    static inline bool Execute( void* i_dstBuffer, void* i_srcBuffer, size_t i_numBytes )
    {
        TOY_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK_ERROR( cudaMemcpy( i_dstBuffer, i_srcBuffer, i_numBytes, cudaMemcpyHostToDevice ) );
    }
};

/// Specialization for synchronous cuda -> host copy.
template <>
struct Copy< Cuda, Host >
{
    static inline bool Execute( void* i_dstBuffer, void* i_srcBuffer, size_t i_numBytes )
    {
        TOY_ASSERT( o_buffer != nullptr );
        return CUDA_CHECK_ERROR( cudaMemcpy( i_dstBuffer, i_srcBuffer, i_numBytes, cudaMemcpyDeviceToHost ) );
    }
};

TOY_NS_CLOSE
