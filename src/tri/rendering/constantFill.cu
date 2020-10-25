#include <gm/types/vec3f.h>

#include <tri/tri.h>

TRI_NS_OPEN

template < typename ValueT >
__global__ void ConstantFill_Kernel( size_t i_numElements, const ValueT i_value, ValueT* o_buffer )
{
    int index = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    if ( index >= i_numElements )
    {
        return;
    }

    o_buffer[ index ] = i_value;
}

// Explicit template definitions for consumed value types.
template __global__ void ConstantFill_Kernel< gm::Vec3f >( size_t i_numElements, const gm::Vec3f i_value, gm::Vec3f* o_buffer );
template __global__ void ConstantFill_Kernel< float >( size_t i_numElements, const float i_value, float* o_buffer );

TRI_NS_CLOSE
