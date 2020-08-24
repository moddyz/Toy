#include <catch2/catch.hpp>

#include <toy/memory/array.h>

using namespace toy;

TEST_CASE( "CudaArray_CopyConstructor" )
{
    Array< float, Cuda > srcArray( 5 );
    Array< float, Cuda > dstArray( srcArray );
    CHECK( dstArray.GetSize() == 5 );
    CHECK( dstArray.GetBuffer() != nullptr );
}

TEST_CASE( "CudaArray_CopyHostToCuda" )
{
    Array< float, Host > srcArray( 5 );
    Array< float, Cuda > dstArray = srcArray;
    CHECK( dstArray.GetSize() == 5 );
    CHECK( dstArray.GetBuffer() != nullptr );
}

TEST_CASE( "CudaArray_CopyCudaToHost" )
{
    Array< float, Cuda > srcArray( 5 );
    Array< float, Host > dstArray = srcArray;
    CHECK( dstArray.GetSize() == 5 );
    CHECK( dstArray.GetBuffer() != nullptr );
}

TEST_CASE( "CudaArray_Resize" )
{
    Array< float, Cuda > array;
    array.Resize( 5 );
    CHECK( array.GetSize() == 5 );
}

