#include <gm/functions/transformPoint.h>
#include <gm/types/mat4f.h>
#include <gm/types/vec3f.h>

#include <toy/toy.h>

TOY_NS_OPEN

__global__ void TransformPoints_Kernel( size_t           i_numPoints,
                                        const gm::Mat4f& i_transform,
                                        const gm::Vec3f* i_points,
                                        gm::Vec3f*       o_points )
{
    int pointIndex = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    if ( pointIndex >= i_numPoints )
    {
        return;
    }

    o_points[ pointIndex ] = gm::TransformPoint( i_transform, i_points[ pointIndex ] );
}

TOY_NS_CLOSE
