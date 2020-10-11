#include <toy/application/viewportWindow.h>
#include <toy/imaging/transformPoints.h>
#include <toy/utils/log.h>

#include <gm/functions/clamp.h>
#include <gm/functions/inverse.h>
#include <gm/types/floatRange.h>
#include <gm/types/vec2f.h>

#include <vector>

TOY_NS_OPEN

class RasterPointsWindow : public ViewportWindow
{
public:
    explicit RasterPointsWindow( const char* i_title, const gm::Vec2i& i_windowSize )
        : ViewportWindow( i_title, i_windowSize )
    {
    }

    virtual void Render( Matrix< gm::Vec3f, Host >& o_image ) override
    {
        // m_points are in world-space. We need to bring them into camera-space.

        // Compute the world to camera matrix.
        const LookAtTransform& lookAtXform   = GetCameraTransform();
        const gm::Mat4f&       cameraToWorld = lookAtXform.GetObjectToWorld();
        gm::Mat4f              worldToCamera;
        TOY_VERIFY( gm::Inverse( cameraToWorld, worldToCamera ) );

        // Perform transformation.
        Array< gm::Vec3f, Host > cameraSpacePoints( m_points.GetSize() );
        TransformPoints< Host >::Execute( worldToCamera, m_points, cameraSpacePoints );
    }

private:
    Array< gm::Vec3f, Host > m_points{gm::Vec3f( -5, -5, 5 ),
                                      gm::Vec3f( 5, 5, 10 ),
                                      gm::Vec3f( 100, 100, 100 ),
                                      gm::Vec3f( 200, 200, 200 )};
};

TOY_NS_CLOSE

int main( int i_argc, char** i_argv )
{
    TOY_LOG_INFO( "[Starting rasterPoints...]\n" );

    toy::RasterPointsWindow window( "Toy: rasterPoints", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
