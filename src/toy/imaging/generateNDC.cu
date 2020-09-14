#include <gm/types/vec2f.h>

#include <toy/memory/cudaError.h>

__global__ void
GenerateNDC_Kernel( int i_numRows, int i_numColumns, float i_rowsInverse, float i_colsInverse, gm::Vec2f* o_pixels )
{
    int xCoord = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    int yCoord = ( blockIdx.y * blockDim.y ) + threadIdx.y;
    if ( xCoord >= i_numColumns || yCoord >= i_numRows )
    {
        return;
    }

    o_pixels[ yCoord * i_numColumns + xCoord ] = gm::Vec2f( xCoord * i_colsInverse, yCoord * i_rowsInverse );
}
