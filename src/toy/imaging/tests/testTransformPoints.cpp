#include <catch2/catch.hpp>

#include <gm/functions/setTranslate.h>

#include <toy/imaging/transformPoints.h>
#include <toy/memory/array.h>

using PointsT     = gm::Vec3f;
using PointsTypes = std::tuple< toy::Array< PointsT, toy::Host >, toy::Array< PointsT, toy::Cuda > >;

TEMPLATE_LIST_TEST_CASE( "TransformPoints", "[template][list]", PointsTypes )
{
    // Perform computation.
    gm::Mat4f transform = gm::Mat4f::Identity();
    gm::SetTranslate( gm::Vec3f( 2.5, 0, 0 ), transform );
    TestType inputPoints = {gm::Vec3f( 0, 0, 0 ),
                            gm::Vec3f( 1, 1, 1 ),
                            gm::Vec3f( 2, 2, 2 ),
                            gm::Vec3f( 3, 3, 3 ),
                            gm::Vec3f( 4, 4, 4 )};
    TestType outputPoints( inputPoints.GetSize() );
    TOY_NS::TransformPoints< TestType::ResidencyType >::Execute( transform, inputPoints, outputPoints );

    // Check results.
    TestType expectedPoints = {gm::Vec3f( 2.5, 0, 0 ),
                               gm::Vec3f( 3.5, 1, 1 ),
                               gm::Vec3f( 4.5, 2, 2 ),
                               gm::Vec3f( 5.5, 3, 3 ),
                               gm::Vec3f( 6.5, 4, 4 )};
    REQUIRE( outputPoints == expectedPoints );
}
