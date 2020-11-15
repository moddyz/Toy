// Implementation of public API.
// All symbols with Tri_ prefix are internal.

#include "tri.h"

#include <unordered_map>
#include <utility>
#include <cassert>
#include <cuda_runtime.h>

/// \class Tri_ObjectContainer
///
/// An object container abstraction, keyed by \ref TriId.
template<typename T>
class Tri_ObjectContainer
{
public:
    // Convenience type definitions.
    using EntryT = std::pair<TriId, T*>;
    using ContainerT = std::unordered_map<TriId, T*>;

    /// Create a new object of type T, specified by a unique identifier.
    ///
    /// \return
    EntryT Create()
    {
        // Allocate a new object and store in container.
        T* object = new T();
        std::pair<typename ContainerT::iterator, bool> insertion =
            m_container.insert(EntryT(m_nextId, object));

        // Check that insertion performed successfully.
        assert(insertion.second);

        // Increment object ID counter.
        m_nextId++;

        return EntryT(m_nextId, object);
    }

    /// Get an existing object of type T.
    T* Get(TriId id) const
    {
        typename ContainerT::const_iterator it = m_container.find(id);
        if (it == m_container.end()) {
            return nullptr;
        }

        return it->second;
    }

private:
    TriId m_nextId = 0;
    ContainerT m_container;
};

/// \class Tri_Context
///
/// Internal TriContext representation.
struct Tri_Context
{
    TriDevice device = TriDevice_Invalid;
};

/// \var s_contexts
///
/// Global container of Tri_Context objects.
static Tri_ObjectContainer<Tri_Context> s_contexts;

/// Select the optimal, preferred device for the current runtime environment.
TriDevice
Tri_SelectPreferredDevice()
{
    int deviceId = TriDevice_Count;
    while (deviceId != TriDevice_CPU) {
        // Decrement.
        deviceId--;

        // Check for device availability.
        if (deviceId == TriDevice_CUDA) {
            int cudaDeviceCount;
            cudaError_t err = cudaGetDeviceCount(&cudaDeviceCount);
            if (err == cudaSuccess && cudaDeviceCount > 0) {
                return (TriDevice)deviceId;
            }
        }
    }

    // Return fallback CPU device.
    assert(deviceId == TriDevice_CPU);
    return (TriDevice)deviceId;
}

TriStatus
TriContextCreatePreferred(TriContext& context)
{
    // Allocate new internal context object.
    typename decltype(s_contexts)::EntryT entry = s_contexts.Create();
    entry.second->device = Tri_SelectPreferredDevice();

    // Populate opaque object ID.
    context.id = entry.first;

    return TriStatus_Success;
}
