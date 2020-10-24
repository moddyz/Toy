#include <gm/types/vec3f.h>

#include <toy/toy.h>

TOY_NS_OPEN

__global__ void ConvertRGBFloatToRGBAUint32_Kernel( size_t i_numPixels, const gm::Vec3f* i_image, uint32_t* o_image )
{
    int pixelIndex = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    if ( pixelIndex >= i_numPixels )
    {
        return;
    }

    const gm::Vec3f& inPixel  = i_image[ pixelIndex ];
    uint8_t*         outPixel = reinterpret_cast< uint8_t* >( &o_image[ pixelIndex ] );
    outPixel[ 0 ]             = static_cast< uint8_t >( 255.999 * inPixel[ 0 ] );
    outPixel[ 1 ]             = static_cast< uint8_t >( 255.999 * inPixel[ 1 ] );
    outPixel[ 2 ]             = static_cast< uint8_t >( 255.999 * inPixel[ 2 ] );
}

TOY_NS_CLOSE
