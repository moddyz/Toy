#include <toy/application/viewportWindow.h>
#include <toy/imaging/extent.h>
#include <toy/imaging/transformPoints.h>
#include <toy/utils/log.h>

#include <gm/functions/clamp.h>
#include <gm/functions/inverse.h>
#include <gm/functions/matrixProduct.h>
#include <gm/functions/orthographicProjection.h>
#include <gm/functions/radians.h>
#include <gm/types/floatRange.h>
#include <gm/types/vec2f.h>

#include <vector>

TOY_NS_OPEN

static gm::Mat4f PerspectiveProjectionFromFrustum( const gm::Vec3fRange& i_view )
{
    // Translate frustum apex to origin.
    gm::Mat4f translateXform = gm::Mat4f::Identity();
    translateXform( 0, 3 )   = -( i_view.Max().X() + i_view.Min().X() ) * 0.5f;
    translateXform( 1, 3 )   = -( i_view.Max().Y() + i_view.Min().Y() ) * 0.5f;
    TOY_LOG_INFO( "translateXform: %s\n", translateXform.GetString().c_str() );

    // Scale depth values in a non-linear range.
    float     c1              = 2.0 * i_view.Max().Z() * i_view.Min().Z() / ( i_view.Min().Z() - i_view.Max().Z() );
    float     c2              = ( i_view.Max().Z() + i_view.Min().Z() ) / ( i_view.Max().Z() - i_view.Min().Z() );
    gm::Mat4f depthScaleXform = gm::Mat4f::Identity();
    depthScaleXform( 2, 2 )   = -c2;
    depthScaleXform( 2, 3 )   = c1;
    depthScaleXform( 3, 2 )   = 1.0f;
    depthScaleXform( 3, 3 )   = 0.0f;
    TOY_LOG_INFO( "depthScaleXform: %s\n", depthScaleXform.GetString().c_str() );

    // Perspective calculation.
    gm::Mat4f projectionXform = gm::Mat4f::Identity();
    projectionXform( 0, 0 )   = i_view.Min().Z();
    projectionXform( 1, 1 )   = i_view.Min().Z();
    TOY_LOG_INFO( "projectionXform: %s\n", projectionXform.GetString().c_str() );

    // Scale viewing window to a 2x2 square (centered at origin).
    gm::Mat4f scaleXform = gm::Mat4f::Identity();
    scaleXform( 0, 0 )   = 2.0f / ( i_view.Max().X() - i_view.Min().X() );
    scaleXform( 1, 1 )   = 2.0f / ( i_view.Max().Y() - i_view.Min().Y() );
    TOY_LOG_INFO( "scaleXform: %s\n", scaleXform.GetString().c_str() );

    return gm::MatrixProduct(
        scaleXform,
        gm::MatrixProduct( projectionXform, gm::MatrixProduct( depthScaleXform, translateXform ) ) );
}

static gm::Mat4f PerspectiveProjection( float i_fieldOfView, float i_aspectRatio, float i_near, float i_far )
{
    // Move the frustum apex to origin.
    float fovRadians = gm::Radians( i_fieldOfView );

    float top    = i_near * tan( fovRadians / 2.0f );
    float bottom = -top;
    float right  = top * i_aspectRatio;
    float left   = -right;
    TOY_LOG_INFO( "top: %f, bottom: %f\n", top, bottom );
    TOY_LOG_INFO( "left: %f, right: %f\n", left, right );

    return PerspectiveProjectionFromFrustum(
        gm::Vec3fRange( gm::Vec3f( left, bottom, i_near ), gm::Vec3f( right, top, i_far ) ) );
}

static gm::Mat4f ClipToRaster( const gm::Vec2i& i_viewportSize )
{
    gm::Mat4f scaleXform = gm::Mat4f::Identity();
    scaleXform( 0, 0 )   = i_viewportSize.X() * 0.5f;
    scaleXform( 1, 1 )   = i_viewportSize.Y() * 0.5f;
    scaleXform( 2, 2 )   = 1;

    gm::Mat4f translateXform = gm::Mat4f::Identity();
    translateXform( 0, 3 )   = i_viewportSize.X() * 0.5f;
    translateXform( 1, 3 )   = i_viewportSize.Y() * 0.5f;

    return gm::MatrixProduct( translateXform, scaleXform );
}

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

        // Compute the world-to-camera-space matrix.
        const LookAtTransform& lookAtXform   = GetCameraTransform();
        const gm::Mat4f&       cameraToWorld = lookAtXform.GetObjectToWorld();
        gm::Mat4f              worldToCamera;
        TOY_VERIFY( gm::Inverse( cameraToWorld, worldToCamera ) );
        TOY_LOG_INFO( "worldToCamera: %s\n", worldToCamera.GetString().c_str() );

        // Compute view matrix.
        gm::Mat4f cameraToClip =
            PerspectiveProjection( 60.0f, ( float ) GetSize().X() / ( float ) GetSize().Y(), 0.1, 1000 );
        TOY_LOG_INFO( "cameraToClip: %s\n", cameraToClip.GetString().c_str() );

        // Clip space -> screen space.
        gm::Mat4f clipToRaster = ClipToRaster( GetSize() );

        // World -> clip space.
        gm::Mat4f worldToClip   = gm::MatrixProduct( cameraToClip, worldToCamera );
        gm::Mat4f worldToRaster = gm::MatrixProduct( clipToRaster, worldToClip );
        TOY_LOG_INFO( "worldToRaster: %s\n", worldToRaster.GetString().c_str() );

        // Screen points.
        Array< gm::Vec3f, Host > screenPoints( m_points.GetSize() );
        TransformPoints< Host >::Execute( worldToRaster, m_points, screenPoints );

        TOY_LOG_INFO( "Points:\n" );
        for ( size_t i = 0; i < screenPoints.GetSize(); ++i )
        {
            TOY_LOG_INFO( "%s\n", screenPoints[ i ].GetString().c_str() );
        }

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

TOY_NS_CLOSE

int main( int i_argc, char** i_argv )
{
    TOY_LOG_INFO( "[Starting rasterizeTriangle...]\n" );

    toy::RasterizeTriangleWindow window( "Toy: rasterTriangle", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
