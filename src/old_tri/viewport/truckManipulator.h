#pragma once

/// \file viewport/truckManipulator.h

#include <tri/viewport/lookAtTransform.h>

#include <gm/functions/clamp.h>
#include <gm/functions/distance.h>
#include <gm/functions/length.h>

TRI_NS_OPEN

/// \class TruckManipulator
///
/// Moves the camera's origin \em and target across the plane formed by its up and right vector.
class TruckManipulator
{
public:
    inline explicit TruckManipulator( LookAtTransform& o_transform, float i_sensitivity = 1.0f )
        : m_transform( o_transform )
        , m_sensitivity( i_sensitivity )
    {
    }

    inline void operator()( const gm::Vec2f& i_mouseDelta )
    {
        gm::Vec3f translation = m_transform.GetNewUp() * m_sensitivity * i_mouseDelta.Y() +
                                m_transform.GetRight() * m_sensitivity * -i_mouseDelta.X();
        m_transform = tri::LookAtTransform( m_transform.GetOrigin() + translation,
                                            m_transform.GetTarget() + translation,
                                            m_transform.GetUp() );
    }

private:
    LookAtTransform& m_transform;
    float            m_sensitivity = 1.0f;
};

TRI_NS_CLOSE
