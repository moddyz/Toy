#include <toy/application/viewportWindow.h>
#include <toy/imaging/transformPoints.h>
#include <toy/utils/log.h>

#include <gm/functions/clamp.h>
#include <gm/functions/inverse.h>
#include <gm/functions/matrixProduct.h>
#include <gm/types/floatRange.h>
#include <gm/types/vec2f.h>

#include <vector>

TOY_NS_OPEN

static gm::Mat4f
OrthographicProjection( float i_left, float i_right, float i_bottom, float i_top, float i_near, float i_far )
{
    // Center volume about origin.
    gm::Mat4f centeringXform = gm::Mat4f::Identity();
    centeringXform( 0, 0 )   = ( i_right + i_left ) * 0.5;
    centeringXform( 0, 1 )   = ( i_top + i_bottom ) * 0.5;
    centeringXform( 0, 2 )   = ( i_far + i_near ) * 0.5;

    // View scaling.
    gm::Mat4f scalingXform = gm::Mat4f::Identity();
    centeringXform( 0, 0 ) = 2.0f / ( i_right + i_left );
    centeringXform( 1, 1 ) = 2.0f / ( i_top + i_bottom );
    centeringXform( 2, 2 ) = 2.0f / ( i_far + i_near );

    return gm::MatrixProduct( scalingXform, centeringXform );
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
        // m_points are in world-space. We need to bring them into camera-space.

        // Compute the world-to-camera-space matrix.
        const LookAtTransform& lookAtXform   = GetCameraTransform();
        const gm::Mat4f&       cameraToWorld = lookAtXform.GetObjectToWorld();
        gm::Mat4f              worldToCamera;
        TOY_VERIFY( gm::Inverse( cameraToWorld, worldToCamera ) );

        // Compute view matrix.
        gm::Mat4f cameraToScreen = OrthographicProjection( -2, 2, -2, 2, -2, 2 );

        // World -> screen space.
        gm::Mat4f worldToScreen = gm::MatrixProduct( cameraToScreen, worldToCamera );

        // Screen points.
        Array< gm::Vec3f, Host > screenPoints( m_points.GetSize() );
        TransformPoints< Host >::Execute( worldToScreen, m_points, screenPoints );
    }

private:
    Array< gm::Vec3f, Host > m_points{gm::Vec3f( 0.0f, 0.57735027f, 0.0f ),
                                      gm::Vec3f( -0.5f, -0.28867513f, 0.0f ),
                                      gm::Vec3f( 0.5f, -0.28867513f, 0.0f )};
};

TOY_NS_CLOSE

int main( int i_argc, char** i_argv )
{
    TOY_LOG_INFO( "[Starting rasterizeTriangle...]\n" );

    toy::RasterizeTriangleWindow window( "Toy: rasterTriangle", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
