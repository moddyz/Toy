#include <toy/imaging/writeImage.h>

#include <gm/types/intRange.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <vector>

TOY_NS_OPEN

bool WriteRGBImage( const Matrix< gm::Vec3f, Host >& i_imageBuffer, const std::string& i_filePath )
{
    size_t                 numChannels = 3;
    std::vector< uint8_t > pixels( i_imageBuffer.GetRows() * i_imageBuffer.GetColumns() * numChannels );

    for ( int yCoord = i_imageBuffer.GetRows() - 1; yCoord >= 0; yCoord-- )
    {
        for ( int xCoord : gm::IntRange( 0, i_imageBuffer.GetColumns() ) )
        {
            const gm::Vec3f& pixel = i_imageBuffer( yCoord, xCoord );

            int pixelOffset           = ( yCoord * i_imageBuffer.GetColumns() + xCoord ) * numChannels;
            pixels[ pixelOffset ]     = static_cast< uint8_t >( 255.999 * pixel[ 0 ] );
            pixels[ pixelOffset + 1 ] = static_cast< uint8_t >( 255.999 * pixel[ 1 ] );
            pixels[ pixelOffset + 2 ] = static_cast< uint8_t >( 255.999 * pixel[ 2 ] );
        }
    }

    return stbi_write_jpg( i_filePath.c_str(),
                           i_imageBuffer.GetColumns(),
                           i_imageBuffer.GetRows(),
                           numChannels,
                           pixels.data(),
                           /* quality */ 100 );
}

TOY_NS_CLOSE

