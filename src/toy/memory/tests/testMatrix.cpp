#include <catch2/catch.hpp>

#include <toy/memory/matrix.h>

using ValueT = float;

using TestUnifiedTypes     = std::tuple< toy::Matrix< ValueT, toy::Host >, toy::Matrix< ValueT, toy::CUDA > >;
using TestIndexAccessTypes = std::tuple< toy::Matrix< ValueT, toy::Host > >;

TEMPLATE_LIST_TEST_CASE( "Matrix_DefaultConstructor", "[template][list]", TestUnifiedTypes )
{
    TestType matrix;
    CHECK( matrix.GetSize() == 0 );
    CHECK( matrix.GetRows() == 0 );
    CHECK( matrix.GetColumns() == 0 );
    CHECK( matrix.GetBuffer() == nullptr );
}

TEMPLATE_LIST_TEST_CASE( "Matrix_SizeConstructor", "[template][list]", TestUnifiedTypes )
{
    TestType matrix( 4, 5 );
    CHECK( matrix.GetRows() == 4 );
    CHECK( matrix.GetColumns() == 5 );
    CHECK( matrix.GetSize() == 20 );
    CHECK( matrix.GetBuffer() != nullptr );
}

TEMPLATE_LIST_TEST_CASE( "Matrix_SizeValueConstructor", "[template][list]", TestIndexAccessTypes )
{
    TestType matrix( 4, 5, 1.0f );
    CHECK( matrix.GetSize() == 20 );
    CHECK( matrix.GetBuffer() != nullptr );
    for ( size_t i = 0; i < 20; ++i )
    {
        CHECK( matrix[ i ] == 1.0f );
    }
}

TEMPLATE_LIST_TEST_CASE( "Matrix_CopyConstructor", "[template][list]", TestIndexAccessTypes )
{
    TestType srcMatrix( 4, 5, 1.0f );
    TestType dstMatrix( srcMatrix );
    CHECK( dstMatrix.GetSize() == 20 );
    CHECK( dstMatrix.GetBuffer() != nullptr );
    CHECK( dstMatrix.GetBuffer() != srcMatrix.GetBuffer() );
    for ( size_t i = 0; i < 20; ++i )
    {
        CHECK( srcMatrix[ i ] == 1.0f );
        CHECK( dstMatrix[ i ] == 1.0f );
    }
}

TEMPLATE_LIST_TEST_CASE( "Matrix_GetSize", "[template][list]", TestUnifiedTypes )
{
    TestType matrix;
    CHECK( matrix.GetSize() == 0 );
    TestType matrixB( 2, 3 );
    CHECK( matrixB.GetSize() == 6 );
}

TEMPLATE_LIST_TEST_CASE( "Matrix_IsEmpty", "[template][list]", TestUnifiedTypes )
{
    TestType matrix;
    CHECK( matrix.IsEmpty() );

    TestType matrixB( 2, 3 );
    CHECK( !matrixB.IsEmpty() );
}

TEMPLATE_LIST_TEST_CASE( "Matrix_Resize", "[template][list]", TestUnifiedTypes )
{
    TestType matrix;
    CHECK( matrix.Resize( 3, 4 ) );
    CHECK( matrix.GetSize() == 12 );
    CHECK( matrix.GetBuffer() != nullptr );
}

TEMPLATE_LIST_TEST_CASE( "Matrix_ResizeValue", "[template][list]", TestIndexAccessTypes )
{
    TestType matrix;
    matrix.Resize( 4, 5, 2.0f );
    CHECK( matrix.GetSize() == 20 );
    CHECK( matrix.GetBuffer() != nullptr );
    for ( size_t i = 0; i < 20; ++i )
    {
        CHECK( matrix[ i ] == 2.0f );
    }
}

TEMPLATE_LIST_TEST_CASE( "Matrix_MatrixElementAccess", "[template][list]", TestIndexAccessTypes )
{
    TestType matrix;
    matrix.Resize( 2, 2, 2.0f );
    for ( size_t i = 0; i < 4; ++i )
    {
        CHECK( matrix[ i ] == 2.0f );
    }

    matrix( 1, 0 ) = 3.0f;
    matrix( 1, 1 ) = 4.0f;

    CHECK( matrix( 0, 0 ) == 2.0f );
    CHECK( matrix( 0, 1 ) == 2.0f );
    CHECK( matrix( 1, 0 ) == 3.0f );
    CHECK( matrix( 1, 1 ) == 4.0f );
}

TEMPLATE_LIST_TEST_CASE( "Matrix_Clear", "[template][list]", TestUnifiedTypes )
{
    TestType matrix( 4, 5 );
    CHECK( matrix.GetSize() == 20 );
    CHECK( matrix.GetBuffer() != nullptr );

    matrix.Clear();
    CHECK( matrix.GetSize() == 0 );
    CHECK( matrix.GetBuffer() == nullptr );
}
