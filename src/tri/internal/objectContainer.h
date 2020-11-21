#pragma once

#include "../tri.h"

#include <unordered_map>
#include <utility>
#include <cassert>

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
    template< typename ClassT >
    EntryT Create()
    {
        TriId objectId = m_nextId++;

        // Allocate a new object and store in container.
        T* object = new ClassT();
        std::pair<typename ContainerT::iterator, bool> insertion =
            m_container.insert(EntryT(objectId, object));

        // Check that insertion performed successfully.
        assert(insertion.second);

        return EntryT(objectId, object);
    }

    /// Delete an existing object referred by \p id.
    bool Delete(TriId id)
    {
        typename ContainerT::iterator it = m_container.find(id);
        if (it == m_container.end()) {
            return false;
        }

        delete it->second;
        m_container.erase(it);

        return true;
    }

    /// Get an existing object by \p id.
    ///
    /// \return Pointer to the object referred by \p id.
    /// \retval nullptr If no object is referred by \p id.
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
