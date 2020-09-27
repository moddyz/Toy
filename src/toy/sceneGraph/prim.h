#pragma once

#include <toy/toy.h>

#include <memory>
#include <string>
#include <vector>

TOY_NS_OPEN

class Prim;

/// \typedef PrimPtr
///
/// Shared pointer to a primitive.
using PrimPtr = std::shared_ptr< Prim >;

/// \class Prim
///
/// A single object in the scene.
class Prim
{
public:
    /// Create a named prim.
    explicit Prim( const std::string& i_name );

    /// Get the name of this prim.
    const std::string& GetName() const
    {
        return m_name;
    }

    /// Get the identifying path of the current prim.
    std::string GetPath() const;

    /// Get the parent primitive.
    ///
    /// If this is the root primitive of the scene, then its parent is invalid (nullptr).
    ///
    /// \return The pointer to the parent primitive.
    PrimPtr GetParent() const
    {
        return m_parent;
    }

    /// Get all the child prims.
    const std::vector< PrimPtr >& GetChildren() const
    {
        return m_children;
    }

    /// Add a child prim.
    ///
    /// \return The newly added child prim.
    PrimPtr AddChild( PrimPtr i_prim )
    {
        m_children.push_back( i_prim );
        return i_prim;
    }

private:
    std::string            m_name   = "default";
    PrimPtr                m_parent = nullptr;
    std::vector< PrimPtr > m_children;
};

TOY_NS_CLOSE
