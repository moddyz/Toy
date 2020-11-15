// Implementation of public API.
// All symbols with Tri_ prefix are internal.

#include "tri.h"

#include "internal/context.h"

TriStatus
TriContextCreatePreferred(TriContext& context)
{
    TriDevice device = Tri_SelectPreferredDevice();
    return Tri_ContextCreate(context, device);
}

TriStatus
TriContextCreate(TriContext& context, TriDevice requestedDevice)
{
    // Check for device availability.
    if (requestedDevice == TriDevice_CUDA && !Tri_IsCUDASupported()) {
        return TriStatus_UnsupportedDevice;
    }

    return Tri_ContextCreate(context, requestedDevice);
}

TriStatus
TriContextDestroy(TriContext& context)
{
    return Tri_ContextDestroy(context);
}

TriStatus
TriContextGetDevice(const TriContext& context, TriDevice& device)
{
    Tri_Context* ctx = Tri_ContextGet(context.id);
    if ( ctx == nullptr )
    {
        return TriStatus_ObjectNotFound;
    }

    device = ctx->device;
    return TriStatus_Success;
}
