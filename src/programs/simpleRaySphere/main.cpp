#include <cxxopts.hpp>

#include <toy/application/window.h>
#include <toy/imaging/convert.h>
#include <toy/imaging/extent.h>
#include <toy/memory/matrix.h>
#include <toy/model/camera.h>
#include <toy/model/ray.h>
#include <toy/utils/log.h>

#include <gm/functions/linearInterpolation.h>
#include <gm/functions/raySphereIntersection.h>
#include <gm/types/vec2iRange.h>
#include <gm/types/vec3f.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

class SimpleRaySphereWindow : public toy::Window
{
public:
    explicit SimpleRaySphereWindow( const char* i_title, const gm::Vec2i& i_dimensions )
        : toy::Window( i_title, i_dimensions )
        , m_camera( /* origin */ gm::Vec3f( 0, 0, 0 ),
                    /* lookAt */ gm::Vec3f( 0, 0, 1 ),
                    /* viewUp */ gm::Vec3f( 0, 1, 0 ),
                    /* verticalFov */ 90.0f,
                    /* aspectRatio */ ( float ) i_dimensions.X() / float( i_dimensions.Y() ) )
    {
    }

protected:
    virtual void Render( uint32_t* o_frameData ) override
    {
        // Cast a ray per pixel to compute the color.
        for ( gm::Vec2i coord : toy::GetImageExtent( m_image ) )
        {
            // Compute normalised viewport coordinates (values between 0 and 1).
            float u = float( coord.X() ) / m_image.GetColumns();
            float v = float( coord.Y() ) / m_image.GetRows();

            toy::Ray ray( m_camera.Origin(),
                          m_camera.ViewportBottomLeft() + ( u * m_camera.ViewportHorizontal() ) +
                              ( v * m_camera.ViewportVertical() ) - m_camera.Origin() );

            // Normalize the direction of the ray.
            ray.Direction() = gm::Normalize( ray.Direction() );

            m_image( coord.Y(), coord.X() ) = _ShadePixel( ray );
        }

        ConvertImageVec3fToUint32( m_image, m_texture );
        CUDA_CHECK( cudaMemcpy( o_frameData, m_texture.GetBuffer(), m_texture.GetByteSize(), cudaMemcpyHostToDevice ) );
    }

    virtual void OnResize( const gm::Vec2i& i_dimensions ) override
    {
        toy::Window::OnResize( i_dimensions );
        m_image.Resize( i_dimensions.Y(), i_dimensions.X() );
    }

    virtual void OnKeyPress( int i_key, int i_action, int i_modifiers ) override
    {
        TOY_LOG_DEBUG( "OnKeyPress: %i, %i, %i\n", i_key, i_action, i_modifiers );

        switch ( i_key )
        {
        case GLFW_KEY_UP:
            m_camera.Origin().Z() += 0.1;
            break;
        case GLFW_KEY_DOWN:
            m_camera.Origin().Z() -= 0.1;
            break;
        }
    }

private:
    static gm::Vec3f _ShadePixel( const toy::Ray& i_ray )
    {
        // Test for sphere intersection (hard-coded placement of the sphere)
        gm::FloatRange intersections;
        if ( gm::RaySphereIntersection( gm::Vec3f( 0, 0, 1.0 ),
                                        0.5,
                                        i_ray.Origin(),
                                        i_ray.Direction(),
                                        intersections ) > 0 )
        {
            return gm::Vec3f( 1, 0, 0 );
        }

        // Compute background color, by interpolating between two colors with the weight as the function of the ray
        // direction.
        float weight = 0.5f * i_ray.Direction().Y() + 1.0;
        return gm::LinearInterpolation( gm::Vec3f( 1.0, 1.0, 1.0 ), gm::Vec3f( 0.5, 0.7, 1.0 ), weight );
    }

    toy::Matrix< gm::Vec3f, toy::Host > m_image;
    toy::Matrix< uint32_t, toy::Host >  m_texture;
    toy::Camera                         m_camera;
};

int main( int i_argc, char** i_argv )
{
    TOY_LOG_INFO( "[Starting simpleRaySphere...]\n" );

    SimpleRaySphereWindow window( "Toy: simpleRaySphere", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
