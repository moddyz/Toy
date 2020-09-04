#pragma once

/// \file utils/hostDevice.h

/// \def TOY_HOST_DEVICE
///
/// Definition to allow functions to be utilized in both host and CUDA device code.
#if defined( __CUDACC__ )
#define TOY_HOST_DEVICE __host__ __device__
#else
#define TOY_HOST_DEVICE
#endif
