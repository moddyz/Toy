#pragma once

#include <toy/memory/array.h>

#include <gm/types/vec2f.h>
#include <gm/types/vec3f.h>

TOY_NS_OPEN

/// \class TriangleMesh
///
/// A mesh described \em completely by triangular faces.
template < Residency ResidencyT >
class TriangleMesh final
{
public:
    Array< gm::Vec3f, ResidencyT > m_positions;
    Array< gm::Vec3f, ResidencyT > m_normals;
    Array< gm::Vec2f, ResidencyT > m_uvs;
    Array< uint32_t, ResidencyT >  m_indices;
};

TOY_NS_CLOSE
