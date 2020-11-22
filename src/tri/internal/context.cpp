#include "context.h"
#include "objectContainer.h"

#include <cuda_runtime.h>
#include <cassert>

/// \var s_contexts
///
/// Global internal container of Tri_Context objects.
///
/// TODO: Exert explicit control over lifetime of this container.
static Tri_ObjectContainer<Tri_Context> s_contexts;

bool
Tri_IsCUDASupported()
{
    int cudaDeviceCount;
    cudaError_t err = cudaGetDeviceCount(&cudaDeviceCount);
    return err == cudaSuccess && cudaDeviceCount > 0;
}

TriDevice
Tri_SelectPreferredDevice()
{
    int deviceId = TriDevice_Count;
    while (deviceId != TriDevice_CPU) {
        // Decrement.
        deviceId--;

        // Check for device availability.
        if (deviceId == TriDevice_CUDA && Tri_IsCUDASupported()) {
            return (TriDevice)deviceId;
        }
    }

    // Return fallback CPU device.
    assert(deviceId == TriDevice_CPU);
    return (TriDevice)deviceId;
}

TriStatus
Tri_ContextCreate(TriContext& context, TriDevice device)
{
    // Allocate new internal context object.
    typename decltype(s_contexts)::EntryT entry =
        s_contexts.Create<Tri_Context>();
    entry.second->device = device;

    // Populate opaque object ID.
    context.id = entry.first;

    return TriStatus_Success;
}

TriStatus
Tri_ContextDestroy(TriContext& context)
{
    if (s_contexts.Delete(context.id)) {
        context = TriContext();
        return TriStatus_Success;
    } else {
        return TriStatus_ObjectNotFound;
    }
}

Tri_Context*
Tri_ContextGet(TriId id)
{
    return s_contexts.Get(id);
}
