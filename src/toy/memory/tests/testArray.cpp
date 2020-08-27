#include <catch2/catch.hpp>

#include <toy/memory/array.h>

using ValueT = float;

using TestUnifiedTypes     = std::tuple< toy::Array< ValueT, toy::Host >, toy::Array< ValueT, toy::Cuda > >;
using TestIndexAccessTypes = std::tuple< toy::Array< ValueT, toy::Host > >;

TEMPLATE_LIST_TEST_CASE( "Array_DefaultConstructor", "[template][list]", TestUnifiedTypes )
{
    TestType array;
    CHECK( array.GetSize() == 0 );
    CHECK( array.GetBuffer() == nullptr );
}

TEMPLATE_LIST_TEST_CASE( "Array_SizeConstructor", "[template][list]", TestUnifiedTypes )
{
    TestType array( 5 );
    CHECK( array.GetSize() == 5 );
    CHECK( array.GetBuffer() != nullptr );
}

TEMPLATE_LIST_TEST_CASE( "Array_SizeValueConstructor", "[template][list]", TestIndexAccessTypes )
{
    TestType array( 5, 1.0f );
    CHECK( array.GetSize() == 5 );
    CHECK( array.GetBuffer() != nullptr );
    for ( size_t i = 0; i < 5; ++i )
    {
        CHECK( array[ i ] == 1 );
    }
}

TEMPLATE_LIST_TEST_CASE( "Array_CopyConstructor", "[template][list]", TestIndexAccessTypes )
{
    TestType srcArray( 5, 1.0f );
    TestType dstArray( srcArray );
    CHECK( dstArray.GetSize() == 5 );
    CHECK( dstArray.GetBuffer() != nullptr );
    CHECK( dstArray.GetBuffer() != srcArray.GetBuffer() );
    for ( size_t i = 0; i < 5; ++i )
    {
        CHECK( srcArray[ i ] == 1 );
        CHECK( dstArray[ i ] == 1 );
    }
}

TEMPLATE_LIST_TEST_CASE( "Array_GetSize", "[template][list]", TestUnifiedTypes )
{
    TestType array;
    CHECK( array.GetSize() == 0 );

    TestType arrayB( 5 );
    CHECK( arrayB.GetSize() == 5 );
}

TEMPLATE_LIST_TEST_CASE( "Array_IsEmpty", "[template][list]", TestUnifiedTypes )
{
    TestType array;
    CHECK( array.IsEmpty() );

    TestType arrayB( 5 );
    CHECK( !arrayB.IsEmpty() );
}

TEMPLATE_LIST_TEST_CASE( "Array_Resize", "[template][list]", TestUnifiedTypes )
{
    TestType array;
    CHECK( array.Resize( 5 ) );
    CHECK( array.GetSize() == 5 );
    CHECK( array.GetBuffer() != nullptr );
}

TEMPLATE_LIST_TEST_CASE( "Array_ResizeValue", "[template][list]", TestIndexAccessTypes )
{
    TestType array;
    array.Resize( 5, 2.0f );
    CHECK( array.GetSize() == 5 );
    CHECK( array.GetBuffer() != nullptr );
    for ( size_t i = 0; i < 5; ++i )
    {
        CHECK( array[ i ] == 2.0f );
    }

    // Resize again,
    array.Resize( 7, 3.0f );
    for ( size_t i = 0; i < 5; ++i )
    {
        CHECK( array[ i ] == 2.0f );
    }
    for ( size_t i = 5; i < 7; ++i )
    {
        CHECK( array[ i ] == 3.0f );
    }
}

TEMPLATE_LIST_TEST_CASE( "Array_Clear", "[template][list]", TestUnifiedTypes )
{
    TestType array( 5 );
    CHECK( array.GetSize() == 5 );
    CHECK( array.GetBuffer() != nullptr );

    array.Clear();
    CHECK( array.GetSize() == 0 );
    CHECK( array.GetBuffer() == nullptr );
}
