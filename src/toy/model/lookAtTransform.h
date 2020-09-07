#pragma once

/// \file model/lookAtTransform.h
///
/// A transformation with an origin (position), look at position, and up vector.
///
/// This transformation is useful for placing a camera (camera-space) in the scene (world-space).
/// The inverse can place world-space scene primitives into camera-space.

#include <toy/toy.h>

#include <gm/functions/lookAt.h>

TOY_NS_OPEN

class LookAtTransform
{
public:
    inline explicit LookAtTransform( const gm::Vec3f& i_origin, const gm::Vec3f& i_target, const gm::Vec3f& i_up )
        : m_origin( i_origin )
        , m_target( i_target )
        , m_up( i_up )
    {
        m_objectToWorld = gm::LookAt( m_origin, m_target, m_up );
    }

    inline const gm::Vec3f& GetOrigin() const
    {
        return m_origin;
    }

    inline const gm::Vec3f& GetTarget() const
    {
        return m_target;
    }

    inline const gm::Vec3f& GetUp() const
    {
        return m_up;
    }

    // Computed params

    inline const gm::Mat4f& GetObjectToWorld() const
    {
        return m_objectToWorld;
    }

    inline gm::Vec3f GetRight() const
    {
        return gm::Vec3f( m_objectToWorld( 0, 0 ), m_objectToWorld( 1, 0 ), m_objectToWorld( 2, 0 ) );
    }

    inline gm::Vec3f GetForward() const
    {
        return gm::Vec3f( m_objectToWorld( 0, 2 ), m_objectToWorld( 1, 2 ), m_objectToWorld( 2, 2 ) );
    }

    inline gm::Vec3f GetNewUp() const
    {
        return gm::Vec3f( m_objectToWorld( 0, 1 ), m_objectToWorld( 1, 1 ), m_objectToWorld( 2, 1 ) );
    }

private:
    // Input parameters.
    gm::Vec3f m_origin;
    gm::Vec3f m_target;
    gm::Vec3f m_up;

    // Computed parameters.
    gm::Mat4f m_objectToWorld;
};

TOY_NS_CLOSE
