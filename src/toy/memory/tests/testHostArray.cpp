#include <catch2/catch.hpp>

#include <toy/memory/array.h>

using namespace toy;

TEST_CASE( "HostArray_Resize" )
{
    Array< float, Host > array;
    array.Resize( 5 );
    CHECK( array.GetSize() == 5 );
}
