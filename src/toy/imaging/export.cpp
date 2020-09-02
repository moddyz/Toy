#include <toy/imaging/convert.h>
#include <toy/imaging/export.h>
#include <toy/imaging/extent.h>
#include <toy/imaging/flip.h>

#include <gm/types/intRange.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <vector>

TOY_NS_OPEN

bool ExportJpeg( const Matrix< gm::Vec3f, Host >& i_image, const std::string& i_filePath )
{
    Matrix< gm::Vec3f, Host > flipped( i_image.GetRows(), i_image.GetColumns() );
    FlipImage( FlipAxis_Y, i_image, flipped );

    size_t                 numChannels = 3;
    std::vector< uint8_t > pixels( flipped.GetRows() * flipped.GetColumns() * numChannels );
    for ( gm::Vec2i coord : GetImageExtent( flipped ) )
    {
        int              pixelOffset = ( coord.Y() * flipped.GetColumns() + coord.X() ) * numChannels;
        const gm::Vec3f& inPixel     = i_image( coord.Y(), coord.X() );
        pixels[ pixelOffset ]        = static_cast< uint8_t >( 255.999 * inPixel[ 0 ] );
        pixels[ pixelOffset + 1 ]    = static_cast< uint8_t >( 255.999 * inPixel[ 1 ] );
        pixels[ pixelOffset + 2 ]    = static_cast< uint8_t >( 255.999 * inPixel[ 2 ] );
    }

    return stbi_write_jpg( i_filePath.c_str(),
                           flipped.GetColumns(),
                           flipped.GetRows(),
                           numChannels,
                           pixels.data(),
                           /* quality */ 100 );
}

TOY_NS_CLOSE
