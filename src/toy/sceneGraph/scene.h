#pragma once

#include <toy/sceneGraph/prim.h>

TOY_NS_OPEN

/// \class Scene
///
/// Container of scene objects (primitives).
class Scene
{
public:
    /// Default constructor.
    Scene();

    /// Get the root prim.
    PrimPtr GetRootPrim() const;

private:
    PrimPtr m_root;
};

TOY_NS_CLOSE
