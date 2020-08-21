#pragma once

/// \file cuda/error.h
///
/// A set of useful utilities for error handling in CUDA programming.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <toy/toy.h>

/// \macro CUDA_ERROR_CONTINUE
///
/// Check the error status.  On non-success, log to stderr and continue exceution.
#define CUDA_ERROR_CONTINUE( val )                                                                                     \
    TOY_NS::_CudaCheckError< _CudaErrorSeverity::Continue >( ( val ), #val, __FILE__, __LINE__ )

/// \macro CUDA_ERROR_FATAL
///
/// Check the error status.  On non-success, log to stderr and exit the program.
#define CUDA_ERROR_FATAL( val )                                                                                        \
    TOY_NS::_CudaCheckError< _CudaErrorSeverity::Fatal >( ( val ), #val, __FILE__, __LINE__ )

TOY_NS_OPEN

/// \enum CudaErrorSeverity
///
/// Severity of CUDA error.
enum class _CudaErrorSeverity : char
{
    Continue = 0,
    Fatal    = 1
};

/// Not intended to be used directly - use \ref CUDA_ERROR_FATAL and \ref CUDA_ERROR_CONTINUE instead.
template < CudaErrorSeverity ErrorSeverityValue >
void _CudaCheckError( cudaError_t i_error, const char* i_function, const char* i_file, int i_line )
{
    if ( i_error != cudaSuccess )
    {
        fprintf( stderr,
                 "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                 i_file,
                 i_line,
                 static_cast< unsigned int >( i_error ),
                 cudaGetErrorName( i_error ),
                 i_function );

        if constexpr ( ErrorSeverityValue == CudaErrorSeverity::Fatal )
        {
            exit( EXIT_FAILURE );
        }
    }
}

TOY_NS_CLOSE
