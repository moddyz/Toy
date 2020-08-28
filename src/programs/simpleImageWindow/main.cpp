#include <cxxopts.hpp>

#include <toy/memory/matrix.h>
#include <toy/present/window.h>
#include <toy/utils/log.h>

#include <gm/types/vec2iRange.h>
#include <gm/types/vec3f.h>

class SimpleImageWindow : public toy::Window
{
public:
    explicit SimpleImageWindow( const char* i_title, const gm::Vec2i& i_dimensions )
        : toy::Window( i_title, i_dimensions )
    {
    }

protected:
    virtual void Render() override
    {
        gm::Vec2iRange bounds( gm::Vec2i( 0, 0 ), gm::Vec2i( m_imageBuffer.GetColumns(), m_imageBuffer.GetRows() ) );
        for ( gm::Vec2i coord : bounds )
        {
            m_imageBuffer( coord.Y(), coord.X() ) =
                gm::Vec3f( ( float ) coord.X() / ( float ) m_imageBuffer.GetColumns(),
                           ( float ) coord.Y() / ( float ) m_imageBuffer.GetRows(),
                           0.0f );
        }
    }

    virtual void GetImage( toy::Matrix< uint32_t, toy::Host >& o_image ) override
    {
        gm::Vec2iRange bounds( gm::Vec2i( 0, 0 ), gm::Vec2i( m_imageBuffer.GetColumns(), m_imageBuffer.GetRows() ) );
        for ( gm::Vec2i coord : bounds )
        {
            const gm::Vec3f& inPixel = m_imageBuffer( coord.Y(), coord.X() );

            uint8_t* outPixel = reinterpret_cast< uint8_t* >( &o_image( coord.Y(), coord.X() ) );
            outPixel[ 0 ]     = static_cast< uint8_t >( 255.999 * inPixel[ 0 ] );
            outPixel[ 1 ]     = static_cast< uint8_t >( 255.999 * inPixel[ 1 ] );
            outPixel[ 2 ]     = static_cast< uint8_t >( 255.999 * inPixel[ 2 ] );
        }
    }

    virtual void OnResize( const gm::Vec2i& i_dimensions ) override
    {
        toy::Window::OnResize( i_dimensions );
        m_imageBuffer.Resize( i_dimensions.Y(), i_dimensions.X() );
    }

private:
    toy::Matrix< gm::Vec3f, toy::Host > m_imageBuffer;
};

int main( int i_argc, char** i_argv )
{
    TOY_LOG_INFO( "[Starting simpleImageWindow...]\n" );

    SimpleImageWindow window( "Toy: simpleImageWindow", gm::Vec2i( 640, 480 ) );
    window.Run();

    return 0;
}
