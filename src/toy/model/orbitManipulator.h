#pragma once

/// \file model/orbitManipulator.h

#include <toy/model/lookAtTransform.h>

#include <gm/functions/setRotate.h>
#include <gm/functions/matrixProduct.h>
#include <gm/functions/transformPoint.h>

TOY_NS_OPEN

/// \class OrbitManipulator
///
/// Manipulator for rotating a camera origin around its target.
class OrbitManipulator
{
public:
    inline explicit OrbitManipulator( LookAtTransform& o_transform, float i_sensitivity = 1.0f )
        : m_transform( o_transform )
        , m_sensitivity( i_sensitivity )
    {
    }

    inline void operator()( const gm::Vec2f& i_mouseDelta )
    {
        // Compute the coordinates of the camera origin, with the target as the coordinate system origin.
        gm::Vec3f targetCentricOrigin = m_transform.GetOrigin() - m_transform.GetTarget();

        // "Pitch" uses the camera right vector as the axis of rotation.
        // The degree of rotation is determined by vertical mouse delta.
        float     pitchDegrees = -i_mouseDelta.Y() * m_sensitivity;
        gm::Mat4f pitchTransform( gm::Mat4f::Identity() );
        gm::SetRotate( pitchDegrees, m_transform.GetRight(), pitchTransform );

        // "Yaw" uses the camera up vector as the axis of rotation.
        // The degree of rotation is determined by horizontal mouse delta.
        float     yawDegrees = i_mouseDelta.X() * m_sensitivity;
        gm::Mat4f yawTransform( gm::Mat4f::Identity() );
        gm::SetRotate( yawDegrees, m_transform.GetNewUp(), yawTransform );

        // Compose the two transforms.
        gm::Mat4f rotationTransform = gm::MatrixProduct( pitchTransform, yawTransform );

        // Perform transformation in target-space, then bring back into world-space.
        gm::Vec3f newTargetCentricOrigin = gm::TransformPoint( rotationTransform, targetCentricOrigin );
        gm::Vec3f newOrigin              = newTargetCentricOrigin + m_transform.GetTarget();

        // The orienting up vector should be (0, 1, 0) in most cases, unless the
        // camera origin is near the region directly above the target.
        gm::Vec3f newUp = gm::Vec3f( 0, 1, 0 );
        if ( gm::DotProduct( gm::Normalize( m_transform.GetTarget() - newOrigin ), newUp ) > 0.99f )
        {
            newUp = m_transform.GetNewUp();
        }

        m_transform = toy::LookAtTransform( newOrigin, m_transform.GetTarget(), m_transform.GetNewUp() );
    }

private:
    LookAtTransform& m_transform;
    float            m_sensitivity = 1.0f;
};

TOY_NS_CLOSE
