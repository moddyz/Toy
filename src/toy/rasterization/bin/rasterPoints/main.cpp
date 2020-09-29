#include <toy/application/viewportWindow.h>
#include <toy/utils/log.h>

#include <gm/types/vec2f.h>
#include <gm/types/floatRange.h>
#include <gm/functions/clamp.h>

#include <vector>

TOY_NS_OPEN

class RasterPointsWindow : public ViewportWindow
{
public:
    explicit RasterPointsWindow( const char* i_title, const gm::Vec2i& i_windowSize )
        : ViewportWindow( i_title, i_windowSize )
    {
        m_points = {
            gm::Vec2f( -5, -5 ),
            gm::Vec2f( 5, 5 ),
            gm::Vec2f( 100, 100 ),
            gm::Vec2f( 200, 200 )
        };
    }

    virtual void Render( Matrix< gm::Vec3f, Host >& o_image ) override
    {
        // Points are already in raster-space.
        for ( gm::Vec2f point : m_points )
        {
            _DrawPointRasterAliased( point, o_image );
        }
    }

private:
    void _DrawPointRasterAliased( const gm::Vec2f& i_point, Matrix< gm::Vec3f, Host >& o_image )
    {
        // Clamp point into raster-space boundaries.
        gm::Vec2f point = gm::Vec2f(
            gm::Clamp( i_point.X(), gm::FloatRange( 0, o_image.GetColumns() - 1 ) ),
            gm::Clamp( i_point.Y(), gm::FloatRange( 0, o_image.GetRows() - 1 ) )
        );

        o_image( std::round( point.Y() ), std::round( point.X() ) ) = gm::Vec3f( 1, 1, 1 );
    }

    std::vector< gm::Vec2f > m_points;
};

TOY_NS_CLOSE

int main( int i_argc, char** i_argv )
{
    TOY_LOG_INFO( "[Starting rasterPoints...]\n" );

    toy::RasterPointsWindow window( "Toy: rasterPoints", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
