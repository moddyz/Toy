#pragma once

/// \file memory/residency.h
///
/// Memory residency.

#include <toy/toy.h>

#include <toy/memory/cudaAllocator.h>
#include <toy/memory/hostAllocator.h>

TOY_NS_OPEN

/// \enum Residency
///
/// The device where memory resides.
enum Residency
{
    Host,
    Cuda
};

/// \struct _GetAllocator
///
/// Template structure for resolving the Allocator class from the Residency enum.
template < Residency ResidencyT >
struct _GetAllocator
{
};

/// Specialization to associate the HostAllocator class with the Host residency enum.
template <>
struct _GetAllocator< Host >
{
    using AllocatorT = HostAllocator;
};

/// Specialization to associate the CudaAllocator class with the Cuda residency enum.
template <>
struct _GetAllocator< Cuda >
{
    using AllocatorT = CudaAllocator;
};

TOY_NS_CLOSE
