#pragma once

#include "../tri.h"

/// \class Tri_Context
///
/// Internal TriContext representation.
struct Tri_Context
{
    TriDevice device{ TriDevice_Uninitialized };
};

/// Check if CUDA is supported by the runtime environment.
bool
Tri_IsCUDASupported();

/// Select the optimal, preferred device for the current runtime environment.
TriDevice
Tri_SelectPreferredDevice();

/// Create a new context.
///
/// \p device is assumed to be supported by the runtime environment.
TriStatus
Tri_ContextCreate(TriContext& context, TriDevice device);

/// Destroys an existing context.
TriStatus
Tri_ContextDestroy(TriContext& context);

/// Fetches an existing context object.
///
/// \retval nullptr If no context is associated with \p id.
Tri_Context*
Tri_ContextGet(TriId id);
