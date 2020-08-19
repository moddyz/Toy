#pragma once

/// \file core/cudaAllocator.h
///
/// CUDA memory allocation strategy.

#include <toy/toy.h>

TOY_NS_OPEN

/// \class CUDAAllocator
///
/// Memory allocation factilies for CUDA.
class CUDAAllocator
{
public:
    /// Allocate a block of managed CUDA memory.
    ///
    /// \param i_size Number of bytes to allocate.
    static inline bool Allocate( size_t i_size )
    {
    }
};

TOY_NS_CLOSE
