#pragma once

/// \file core/export.h
///
/// Cross-platform symbol export definitions.
///
/// Windows platform requires flag for marking as symbol for external linkage.

#include <toy/core/os.h>

#if defined( TOY_WINDOWS )
#    define TOY_EXPORT __declspec( dllexport )
#else
#    define TOY_EXPORT
#endif
