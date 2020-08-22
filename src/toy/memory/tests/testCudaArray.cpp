#include <catch2/catch.hpp>

#include <toy/memory/array.h>

using namespace toy;

TEST_CASE( "CudaArray_Resize" )
{
    Array< float, Cuda > array;
    array.Resize( 5 );
    CHECK( array.GetSize() == 5 );
}
