#include <cxxopts.hpp>

#include <toy/application/window.h>
#include <toy/imaging/convert.h>
#include <toy/imaging/extent.h>
#include <toy/memory/matrix.h>
#include <toy/memory/cudaError.h>
#include <toy/utils/log.h>

#include <gm/types/vec2iRange.h>
#include <gm/types/vec3f.h>

#include <cuda_runtime.h>

class SimpleImageWindow : public toy::Window
{
public:
    explicit SimpleImageWindow( const char* i_title, const gm::Vec2i& i_dimensions )
        : toy::Window( i_title, i_dimensions )
    {
    }

protected:
    virtual void WriteFrame( uint32_t* o_frameData ) override
    {
        for ( gm::Vec2i coord : toy::GetImageExtent( m_image ) )
        {
            m_image( coord.Y(), coord.X() ) = gm::Vec3f( ( float ) coord.X() / ( float ) m_image.GetColumns(),
                                                         ( float ) coord.Y() / ( float ) m_image.GetRows(),
                                                         0.0f );
        }
        ConvertImageVec3fToUint32( m_image, m_texture );
        CUDA_CHECK( cudaMemcpy( o_frameData, m_texture.GetBuffer(), m_texture.GetByteSize(), cudaMemcpyHostToDevice ) );
    }

    virtual void OnResize( const gm::Vec2i& i_dimensions ) override
    {
        m_image.Resize( i_dimensions.Y(), i_dimensions.X() );
        m_texture.Resize( i_dimensions.Y(), i_dimensions.X() );
    }

private:
    toy::Matrix< gm::Vec3f, toy::Host > m_image;
    toy::Matrix< uint32_t, toy::Host >  m_texture;
};

int main( int i_argc, char** i_argv )
{
    TOY_LOG_INFO( "[Starting simpleImageWindow...]\n" );

    SimpleImageWindow window( "Toy: simpleImageWindow", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
