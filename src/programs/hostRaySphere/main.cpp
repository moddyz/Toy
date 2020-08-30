#include <cxxopts.hpp>

#include <toy/app/window.h>
#include <toy/imaging/convert.h>
#include <toy/imaging/extent.h>
#include <toy/memory/matrix.h>
#include <toy/model/camera.h>
#include <toy/model/ray.h>
#include <toy/utils/log.h>

#include <gm/types/vec2iRange.h>
#include <gm/types/vec3f.h>
#include <gm/functions/linearInterpolation.h>

class HostRaySphereWindow : public toy::Window
{
public:
    explicit HostRaySphereWindow( const char* i_title, const gm::Vec2i& i_dimensions )
        : toy::Window( i_title, i_dimensions )
        , m_camera( /* origin */ gm::Vec3f( 0, 0, 0 ),
                    /* lookAt */ gm::Vec3f( 0, 0, 1 ),
                    /* viewUp */ gm::Vec3f( 0, 1, 0 ),
                    /* verticalFov */ 90.0f,
                    /* aspectRatio */ ( float ) i_dimensions.X() / float( i_dimensions.Y() ) )
    {
    }

protected:
    virtual void Render() override
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

            float     weight = 0.5f * ray.Direction().Y() + 1.0;
            gm::Vec3f color = gm::LinearInterpolation( gm::Vec3f( 1.0, 1.0, 1.0 ), gm::Vec3f( 0.5, 0.7, 1.0 ), weight );

            m_image( coord.Y(), coord.X() ) = color;
        }
    }

    virtual void ConvertImageToTexture( toy::Matrix< uint32_t, toy::Host >& o_texture ) override
    {
        toy::ConvertImageVec3fToUint32( m_image, o_texture );
    }

    virtual void OnResize( const gm::Vec2i& i_dimensions ) override
    {
        toy::Window::OnResize( i_dimensions );
        m_image.Resize( i_dimensions.Y(), i_dimensions.X() );
    }

private:
    toy::Matrix< gm::Vec3f, toy::Host > m_image;
    toy::Camera                         m_camera;
};

int main( int i_argc, char** i_argv )
{
    TOY_LOG_INFO( "[Starting hostRaySphere...]\n" );

    HostRaySphereWindow window( "Toy: hostRaySphere", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
