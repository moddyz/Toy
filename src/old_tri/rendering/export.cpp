#include <tri/rendering/export.h>
#include <tri/rendering/flip.h>

#include <gm/types/intRange.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <vector>

TRI_NS_OPEN

bool ExportJpeg( const FrameBuffer< gm::Vec3f, Host >& i_frameBuffer, const std::string& i_filePath )
{
    FrameBuffer< gm::Vec3f, Host > flipped( i_frameBuffer.GetDimensions() );
    FlipImage( FlipAxis_Y, i_frameBuffer, flipped );

    size_t                 numChannels = 3;
    std::vector< uint8_t > pixels( flipped.GetWidth() * flipped.GetHeight() * numChannels );
    for ( gm::Vec3i coord : flipped.GetExtent() )
    {
        int              pixelOffset = ( coord.Y() * flipped.GetWidth() + coord.X() ) * numChannels;
        const gm::Vec3f& inPixel     = i_frameBuffer( coord.X(), coord.Y(), coord.Z() );
        pixels[ pixelOffset ]        = static_cast< uint8_t >( 255.999 * inPixel[ 0 ] );
        pixels[ pixelOffset + 1 ]    = static_cast< uint8_t >( 255.999 * inPixel[ 1 ] );
        pixels[ pixelOffset + 2 ]    = static_cast< uint8_t >( 255.999 * inPixel[ 2 ] );
    }

    return stbi_write_jpg( i_filePath.c_str(),
                           flipped.GetHeight(),
                           flipped.GetWidth(),
                           numChannels,
                           pixels.data(),
                           /* quality */ 100 );
}

TRI_NS_CLOSE
