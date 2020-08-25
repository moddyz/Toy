#include <catch2/catch.hpp>

#include <toy/memory/array.h>

using ValueT                               = float;
static constexpr toy::Residency ResidencyT = toy::Host;

TEST_CASE( "HostArray_DefaultConstructor" )
{
    toy::Array< ValueT, ResidencyT > array;
    CHECK( array.GetSize() == 0 );
    CHECK( array.GetBuffer() == nullptr );
}

TEST_CASE( "HostArray_SizeConstructor" )
{
    toy::Array< ValueT, ResidencyT > array( 5 );
    CHECK( array.GetSize() == 5 );
    CHECK( array.GetBuffer() != nullptr );
}

TEST_CASE( "HostArray_SizeValueConstructor" )
{
    toy::Array< ValueT, ResidencyT > array( 5, 1.0f );
    CHECK( array.GetSize() == 5 );
    CHECK( array.GetBuffer() != nullptr );
    for ( size_t i = 0; i < 5; ++i )
    {
        CHECK( array[ i ] == 1.0f );
    }
}

TEST_CASE( "HostArray_CopyConstructor" )
{
    toy::Array< ValueT, ResidencyT > srcArray( 5, 1.0f );
    toy::Array< ValueT, ResidencyT > dstArray( srcArray );
    CHECK( dstArray.GetSize() == 5 );
    CHECK( dstArray.GetBuffer() != nullptr );
    CHECK( dstArray.GetBuffer() != srcArray.GetBuffer() );
    for ( size_t i = 0; i < 5; ++i )
    {
        CHECK( srcArray[ i ] == 1 );
        CHECK( dstArray[ i ] == 1 );
    }
}

TEST_CASE( "HostArray_Resize" )
{
    toy::Array< ValueT, ResidencyT > array;
    array.Resize( 5 );
    CHECK( array.GetSize() == 5 );
    CHECK( array.GetBuffer() != nullptr );
}

TEST_CASE( "HostArray_Clear" )
{
    toy::Array< ValueT, ResidencyT > array( 5 );
    CHECK( array.GetSize() == 5 );
    CHECK( array.GetBuffer() != nullptr );

    array.Clear();
    CHECK( array.GetSize() == 0 );
    CHECK( array.GetBuffer() == nullptr );
}