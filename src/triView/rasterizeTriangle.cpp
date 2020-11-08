#include "viewportWindow.h"
#include "constantFill.h"
#include "transformPoints.h"

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

class RasterizeTriangleWindow : public ViewportWindow
{
public:
    explicit RasterizeTriangleWindow( const char* i_title, const gm::Vec2i& i_windowSize )
        : ViewportWindow( i_title, i_windowSize )
    {
    }

    virtual void Render( FrameBuffer< gm::Vec3f, CUDA >& o_colorBuffer ) override
    {
    }

private:
    Array< gm::Vec3f, CUDA > m_points{gm::Vec3f( 0.0f, 0.57735027f, 0.0f ),
                                      gm::Vec3f( -0.5f, -0.28867513f, 0.0f ),
                                      gm::Vec3f( 0.5f, -0.28867513f, 0.0f )};
};

int main( int i_argc, char** i_argv )
{
    printf( "[Starting rasterizeTriangle...]\n" );

    RasterizeTriangleWindow window( "Tri: rasterTriangle", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
