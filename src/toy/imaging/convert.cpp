#include <toy/imaging/convert.h>
#include <toy/imaging/extent.h>

TOY_NS_OPEN

void ConvertImageVec3fToUint32( const Matrix< gm::Vec3f, Host >& i_image, Matrix< uint32_t, Host >& o_image )
{
    o_image.Resize( i_image.GetRows(), i_image.GetColumns() );
    for ( gm::Vec2i coord : GetImageExtent( i_image ) )
    {
        const gm::Vec3f& inPixel  = i_image( coord.Y(), coord.X() );
        uint8_t*         outPixel = reinterpret_cast< uint8_t* >( &o_image( coord.Y(), coord.X() ) );
        outPixel[ 0 ]             = static_cast< uint8_t >( 255.999 * inPixel[ 0 ] );
        outPixel[ 1 ]             = static_cast< uint8_t >( 255.999 * inPixel[ 1 ] );
        outPixel[ 2 ]             = static_cast< uint8_t >( 255.999 * inPixel[ 2 ] );
    }
}

TOY_NS_CLOSE
