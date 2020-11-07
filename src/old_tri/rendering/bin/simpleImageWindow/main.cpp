#include <cxxopts.hpp>

#include <tri/application/window.h>
#include <tri/base/log.h>
#include <tri/memory/cudaError.h>
#include <tri/rendering/formatConversion.h>
#include <tri/rendering/frameBuffer.h>

#include <gm/types/vec2iRange.h>
#include <gm/types/vec3f.h>

#include <cuda_runtime.h>

TRI_NS_USING

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
        for ( gm::Vec3i coord : m_colorBuffer.GetExtent() )
        {
            m_colorBuffer( coord ) = gm::Vec3f( ( float ) coord.X() / ( float ) m_colorBuffer.GetWidth(),
                                                ( float ) coord.Y() / ( float ) m_colorBuffer.GetHeight(),
                                                0.0f );
        }
        tri::ConvertRGBFloatToRGBAUint32< tri::Host >::Execute( m_colorBuffer.GetElementCount(),
                                                                m_colorBuffer.GetBuffer(),
                                                                m_texture.GetBuffer() );
        CUDA_CHECK( cudaMemcpy( o_frameData, m_texture.GetBuffer(), m_texture.GetByteSize(), cudaMemcpyHostToDevice ) );
    }

    virtual void OnResize( const gm::Vec2i& i_dimensions ) override
    {
        m_colorBuffer.Resize( gm::Vec3i( i_dimensions.X(), i_dimensions.Y(), 1 ) );
        m_texture.Resize( gm::Vec3i( i_dimensions.X(), i_dimensions.Y(), 1 ) );
    }

private:
    tri::FrameBuffer< gm::Vec3f, tri::Host > m_colorBuffer;
    tri::FrameBuffer< uint32_t, tri::Host >  m_texture;
};

int main( int i_argc, char** i_argv )
{
    TRI_LOG_INFO( "[Starting simpleImageWindow...]\n" );

    SimpleImageWindow window( "Tri: simpleImageWindow", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
