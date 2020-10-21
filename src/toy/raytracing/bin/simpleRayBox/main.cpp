#include <cxxopts.hpp>

#include <toy/application/viewportWindow.h>
#include <toy/imaging/extent.h>
#include <toy/imaging/ray.h>
#include <toy/base/log.h>

#include <gm/functions/linearInterpolation.h>
#include <gm/functions/rayAABBIntersection.h>
#include <gm/functions/transformPoint.h>
#include <gm/functions/transformVector.h>
#include <gm/types/vec2iRange.h>
#include <gm/types/vec3f.h>

TOY_NS_OPEN

class SimpleRayBoxWindow : public ViewportWindow
{
public:
    explicit SimpleRayBoxWindow( const char* i_title, const gm::Vec2i& i_dimensions )
        : ViewportWindow( i_title, i_dimensions )
    {
    }

    virtual void Render( Matrix< gm::Vec3f, Host >& o_image ) override
    {
        // Cast a ray per pixel to compute the color.
        for ( gm::Vec2i coord : GetImageExtent( o_image ) )
        {
            // Compute normalised viewport coordinates (values between 0 and 1).
            float u = float( coord.X() ) / o_image.GetColumns();
            float v = float( coord.Y() ) / o_image.GetRows();

            Ray ray( gm::Vec3f( 0, 0, 0 ),
                     GetCameraView().NearBottomLeft() + ( u * GetCameraView().NearHorizontal() ) +
                         ( v * GetCameraView().NearVertical() ) );

            // Transform camera-space ray into world-space ray.
            ray.Origin() = gm::TransformPoint( GetCameraTransform().GetObjectToWorld(), ray.Origin() );
            ray.Direction() =
                gm::Normalize( gm::TransformVector( GetCameraTransform().GetObjectToWorld(), ray.Direction() ) );

            o_image( coord.Y(), coord.X() ) = _ShadePixel( ray );
        }
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
};

TOY_NS_CLOSE

int main( int i_argc, char** i_argv )
{
    TOY_LOG_INFO( "[Starting simpleRayBox...]\n" );

    toy::SimpleRayBoxWindow window( "Toy: simpleRayBox", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
