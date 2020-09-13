#pragma once

#include <toy/application/window.h>
#include <toy/imaging/convert.h>
#include <toy/imaging/extent.h>
#include <toy/memory/matrix.h>
#include <toy/model/dollyManipulator.h>
#include <toy/model/lookAtTransform.h>
#include <toy/model/orbitManipulator.h>
#include <toy/model/perspectiveView.h>
#include <toy/model/ray.h>
#include <toy/model/truckManipulator.h>
#include <toy/utils/log.h>

#include <gm/functions/linearInterpolation.h>
#include <gm/functions/rayAABBIntersection.h>
#include <gm/functions/transformPoint.h>
#include <gm/functions/transformVector.h>
#include <gm/types/vec2iRange.h>
#include <gm/types/vec3f.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

TOY_NS_OPEN

template< Residency ResidencyT >
class SimpleRayPipelineWindow : public Window
{
public:
    explicit SimpleRayPipelineWindow( const char* i_title, const gm::Vec2i& i_dimensions )
        : Window( i_title, i_dimensions )
        , m_cameraTransform( /* origin */ gm::Vec3f( 0, 0, -5 ),
                             /* target */ gm::Vec3f( 0, 0, 0 ),
                             /* up */ gm::Vec3f( 0, 1, 0 ) )
        , m_cameraView( /* verticalFov */ 90.0f,
                        /* aspectRatio */ ( float ) i_dimensions.X() / float( i_dimensions.Y() ) )
    {
    }

protected:
    virtual void Render( uint32_t* o_frameData ) override
    {
        // Cast a ray per pixel to compute the color.
        for ( gm::Vec2i coord : GetImageExtent( m_image ) )
        {
            // Compute normalised viewport coordinates (values between 0 and 1).
            float u = float( coord.X() ) / m_image.GetColumns();
            float v = float( coord.Y() ) / m_image.GetRows();

            Ray ray( gm::Vec3f( 0, 0, 0 ),
                     m_cameraView.NearBottomLeft() + ( u * m_cameraView.NearHorizontal() ) +
                         ( v * m_cameraView.NearVertical() ) );

            // Transform camera-space ray into world-space ray.
            ray.Origin() = gm::TransformPoint( m_cameraTransform.GetObjectToWorld(), ray.Origin() );
            ray.Direction() =
                gm::Normalize( gm::TransformVector( m_cameraTransform.GetObjectToWorld(), ray.Direction() ) );

            m_image( coord.Y(), coord.X() ) = _ShadePixel( ray );
        }

        ConvertImageVec3fToUint32( m_image, m_texture );
        CUDA_CHECK( cudaMemcpy( o_frameData, m_texture.GetBuffer(), m_texture.GetByteSize(), cudaMemcpyHostToDevice ) );
    }

    virtual void OnResize( const gm::Vec2i& i_dimensions ) override
    {
        m_image.Resize( i_dimensions.Y(), i_dimensions.X() );
    }

    virtual void OnMouseMove( const gm::Vec2f& i_position ) override
    {
        gm::Vec2f mouseDelta = i_position - GetLastMousePosition();

        if ( GetMouseButtonPressed() & MouseButton_Left )
        {
            OrbitManipulator orbitManip( m_cameraTransform );
            orbitManip( mouseDelta );
        }
        else if ( GetMouseButtonPressed() & MouseButton_Middle )
        {
            TruckManipulator truckManip( m_cameraTransform, /* sensitivity */ 0.01f );
            truckManip( mouseDelta );
        }
        else if ( GetMouseButtonPressed() & MouseButton_Right )
        {
            DollyManipulator dollyManip( m_cameraTransform, /* sensitivity */ 0.01f );
            dollyManip( mouseDelta.Y() );
        }
    }

    virtual void OnMouseScroll( const gm::Vec2f& i_offset ) override
    {
        DollyManipulator dollyManip( m_cameraTransform );
        dollyManip( i_offset.Y() );
    }

private:
    static gm::Vec3f _ShadePixel( const Ray& i_ray )
    {
        // Test for box intersection (hard-coded placement of the box)
        gm::FloatRange intersections;
        if ( gm::RayAABBIntersection( i_ray.Origin(),
                                      i_ray.Direction(),
                                      gm::Vec3fRange( gm::Vec3f( -1, -1, -1 ), gm::Vec3f( 1, 1, 1 ) ),
                                      intersections ) > 0 )
        {
            return gm::Vec3f( 1, 0, 0 );
        }

        // Compute background color, by interpolating between two colors with the weight as the function of the ray
        // direction.
        float weight = 0.5f * i_ray.Direction().Y() + 1.0;
        return gm::LinearInterpolation( gm::Vec3f( 1.0, 1.0, 1.0 ), gm::Vec3f( 0.5, 0.7, 1.0 ), weight );
    }

    // Frame buffer.
    Matrix< gm::Vec3f, ResidencyT > m_image;
    Matrix< uint32_t, ResidencyT >  m_texture;

    // Camera.
    LookAtTransform m_cameraTransform;
    PerspectiveView m_cameraView;
};

TOY_NS_CLOSE
