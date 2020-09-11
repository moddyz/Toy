#pragma once

/// \file model/dollyManipulator.h

#include <toy/model/lookAtTransform.h>

#include <gm/functions/clamp.h>
#include <gm/functions/distance.h>
#include <gm/functions/length.h>

TOY_NS_OPEN

/// \class DollyManipulator
///
/// Moves the camera origin directly towards or away from the target.
class DollyManipulator
{
public:
    inline explicit DollyManipulator( LookAtTransform& o_transform, float i_sensitivity = 1.0f )
        : m_transform( o_transform )
        , m_sensitivity( i_sensitivity )
    {
    }

    inline void operator()( float i_delta )
    {
        TOY_LOG_DEBUG( "Delta: %f\n", i_delta );

        // Compute the translation vector.
        gm::Vec3f translation = m_transform.GetForward() * m_sensitivity * i_delta;
        TOY_LOG_DEBUG( "Translation: %s\n", translation.GetString().c_str() );

        // Soften the translation vector _only_ if translation is towards the target.
        gm::Vec3f originToTarget = m_transform.GetTarget() - m_transform.GetOrigin();
        if ( gm::DotProduct( originToTarget, translation ) > 0 )
        {
            float targetOriginDistance = gm::Distance( m_transform.GetOrigin(), m_transform.GetTarget() );
            float translationLength = gm::Clamp( gm::Length( translation ), gm::FloatRange( 0, targetOriginDistance ) );
            float multiplier = 1.0f - ( translationLength / targetOriginDistance );
            translation *= multiplier;
        }

        m_transform =
            toy::LookAtTransform( m_transform.GetOrigin() + translation, m_transform.GetTarget(), m_transform.GetUp() );
    }

private:
    LookAtTransform& m_transform;
    float            m_sensitivity = 1.0f;
};

TOY_NS_CLOSE
