#pragma once

/// \file toy/toy.h
///
/// CUDA-based toy ray tracer.

#include <toy/core/export.h>

/// \def TOY_API
///
/// Mark a symbol for external linkage.
#define TOY_API TOY_EXPORT

/// \def TOY_NS
///
/// The namespace hosting all the symbols in the toy library.
#define TOY_NS toy

/// \def TOY_NS_USING
///
/// Convenience using directive for TOY_NS.
#define TOY_NS_USING using namespace TOY_NS;

/// \def TOY_NS_OPEN
///
/// Used throughout the library to open the toy namespace scope.
#define TOY_NS_OPEN                                                                                                    \
    namespace TOY_NS                                                                                                   \
    {
/// \def TOY_NS_CLOSE
///
/// Used throughout the library to close the toy namespace scope.
#define TOY_NS_CLOSE }
