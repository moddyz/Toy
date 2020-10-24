#include <tri/application/viewportWindow.h>
#include <tri/base/log.h>
#include <tri/imaging/extent.h>
#include <tri/imaging/transformPoints.h>

#include <gm/functions/clamp.h>
#include <gm/functions/inverse.h>
#include <gm/functions/matrixProduct.h>
#include <gm/functions/orthographicProjection.h>
#include <gm/functions/perspectiveProjection.h>
#include <gm/functions/radians.h>
#include <gm/functions/viewportTransform.h>
#include <gm/types/floatRange.h>
#include <gm/types/vec2f.h>

#include <vector>

TRI_NS_OPEN

static bool PointInsideTriangle( const gm::Vec3f& point, const gm::Vec3f triangle[ 3 ] )
{
    float dX   = point[ 0 ] - triangle[ 2 ][ 0 ];
    float dY   = point[ 1 ] - triangle[ 2 ][ 1 ];
    float dX21 = triangle[ 2 ][ 0 ] - triangle[ 1 ][ 0 ];
    float dY12 = triangle[ 1 ][ 1 ] - triangle[ 2 ][ 1 ];
    float D = dY12 * ( triangle[ 0 ][ 0 ] - triangle[ 2 ][ 0 ] ) + dX21 * ( triangle[ 0 ][ 1 ] - triangle[ 2 ][ 1 ] );
    float s = dY12 * dX + dX21 * dY;
    float t = ( triangle[ 2 ][ 1 ] - triangle[ 0 ][ 1 ] ) * dX + ( triangle[ 0 ][ 0 ] - triangle[ 2 ][ 0 ] ) * dY;
    if ( D < 0 )
    {
        return s <= 0 && t <= 0 && s + t >= D;
    }
    else
    {
        return s >= 0 && t >= 0 && s + t <= D;
    }
}

class RasterizeTriangleWindow : public ViewportWindow
{
public:
    explicit RasterizeTriangleWindow( const char* i_title, const gm::Vec2i& i_windowSize )
        : ViewportWindow( i_title, i_windowSize )
    {
    }

    virtual void Render( Matrix< gm::Vec3f, Host >& o_image ) override
    {
        // Clear.
        for ( gm::Vec2i coord : GetImageExtent( o_image ) )
        {
            o_image( coord.Y(), coord.X() ) = gm::Vec3f( 0.0, 0.0, 0.0 );
        }

        // World-space to camera-space.
        const LookAtTransform& lookAtXform   = GetCameraTransform();
        const gm::Mat4f&       cameraToWorld = lookAtXform.GetObjectToWorld();
        gm::Mat4f              worldToCamera;
        TRI_VERIFY( gm::Inverse( cameraToWorld, worldToCamera ) );

        // Compose transformation bringing 3D geometry to the 2D screen.
        gm::Mat4f cameraToClip =
            gm::PerspectiveProjection( /* fov */ 60.0f,
                                       /* aspectRatio */ ( float ) GetSize().X() / ( float ) GetSize().Y(),
                                       0.1,
                                       1000 );
        gm::Mat4f clipToRaster  = gm::ViewportTransform( gm::Vec2f( GetSize().X(), GetSize().Y() ), gm::Vec2f( 0, 0 ) );
        gm::Mat4f worldToClip   = gm::MatrixProduct( cameraToClip, worldToCamera );
        gm::Mat4f worldToRaster = gm::MatrixProduct( clipToRaster, worldToClip );

        // Screen points.
        Array< gm::Vec3f, Host > screenPoints( m_points.GetSize() );
        TransformPoints< Host >::Execute( worldToRaster, m_points, screenPoints );

        for ( gm::Vec2i coord : GetImageExtent( o_image ) )
        {
            if ( PointInsideTriangle( gm::Vec3f( coord.X(), coord.Y(), 0 ), &( screenPoints[ 0 ] ) ) )
            {
                o_image( coord.Y(), coord.X() ) = gm::Vec3f( 1.0, 1.0, 1.0 );
            }
        }
    }

private:
    Array< gm::Vec3f, Host > m_points{gm::Vec3f( 0.0f, 0.57735027f, 0.0f ),
                                      gm::Vec3f( -0.5f, -0.28867513f, 0.0f ),
                                      gm::Vec3f( 0.5f, -0.28867513f, 0.0f )};
};

TRI_NS_CLOSE

int main( int i_argc, char** i_argv )
{
    TRI_LOG_INFO( "[Starting rasterizeTriangle...]\n" );

    tri::RasterizeTriangleWindow window( "Tri: rasterTriangle", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
