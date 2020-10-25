#include <cxxopts.hpp>

#include <tri/application/window.h>
#include <tri/base/log.h>
#include <tri/rendering/extent.h>
#include <tri/rendering/formatConversion.h>
#include <tri/memory/cudaError.h>
#include <tri/memory/matrix.h>

#include <gm/types/vec2iRange.h>
#include <gm/types/vec3f.h>

#include <cuda_runtime.h>

class SimpleImageWindow : public tri::Window
{
public:
    explicit SimpleImageWindow( const char* i_title, const gm::Vec2i& i_dimensions )
        : tri::Window( i_title, i_dimensions )
    {
    }

protected:
    virtual void WriteFrame( uint32_t* o_frameData ) override
    {
        for ( gm::Vec2i coord : tri::GetImageExtent( m_image ) )
        {
            m_image( coord.Y(), coord.X() ) = gm::Vec3f( ( float ) coord.X() / ( float ) m_image.GetColumns(),
                                                         ( float ) coord.Y() / ( float ) m_image.GetRows(),
                                                         0.0f );
        }
        tri::ConvertRGBFloatToRGBAUint32< tri::Host >::Execute( m_image.GetSize(),
                                                                m_image.GetBuffer(),
                                                                m_texture.GetBuffer() );
        CUDA_CHECK( cudaMemcpy( o_frameData, m_texture.GetBuffer(), m_texture.GetByteSize(), cudaMemcpyHostToDevice ) );
    }

    virtual void OnResize( const gm::Vec2i& i_dimensions ) override
    {
        m_image.Resize( i_dimensions.Y(), i_dimensions.X() );
        m_texture.Resize( i_dimensions.Y(), i_dimensions.X() );
    }

private:
    tri::Matrix< gm::Vec3f, tri::Host > m_image;
    tri::Matrix< uint32_t, tri::Host >  m_texture;
};

int main( int i_argc, char** i_argv )
{
    TRI_LOG_INFO( "[Starting simpleImageWindow...]\n" );

    SimpleImageWindow window( "Tri: simpleImageWindow", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
