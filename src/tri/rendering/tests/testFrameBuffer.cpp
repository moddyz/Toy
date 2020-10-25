#include <catch2/catch.hpp>

#include <tri/rendering/frameBuffer.h>

using ValueT = float;

using TestUnifiedTypes     = std::tuple< tri::FrameBuffer< ValueT, tri::Host >, tri::FrameBuffer< ValueT, tri::CUDA > >;
using TestIndexAccessTypes = std::tuple< tri::FrameBuffer< ValueT, tri::Host > >;

TEMPLATE_LIST_TEST_CASE( "FrameBuffer_DefaultConstructor", "[template][list]", TestUnifiedTypes )
{
    TestType frameBuffer;
    CHECK( frameBuffer.GetSize() == 0 );
    CHECK( frameBuffer.GetRows() == 0 );
    CHECK( frameBuffer.GetColumns() == 0 );
    CHECK( frameBuffer.GetBuffer() == nullptr );
}

TEMPLATE_LIST_TEST_CASE( "FrameBuffer_SizeConstructor", "[template][list]", TestUnifiedTypes )
{
    TestType frameBuffer( 4, 5 );
    CHECK( frameBuffer.GetRows() == 4 );
    CHECK( frameBuffer.GetColumns() == 5 );
    CHECK( frameBuffer.GetSize() == 20 );
    CHECK( frameBuffer.GetBuffer() != nullptr );
}

TEMPLATE_LIST_TEST_CASE( "FrameBuffer_SizeValueConstructor", "[template][list]", TestIndexAccessTypes )
{
    TestType frameBuffer( 4, 5, 1.0f );
    CHECK( frameBuffer.GetSize() == 20 );
    CHECK( frameBuffer.GetBuffer() != nullptr );
    for ( size_t i = 0; i < 20; ++i )
    {
        CHECK( frameBuffer[ i ] == 1.0f );
    }
}

TEMPLATE_LIST_TEST_CASE( "FrameBuffer_CopyConstructor", "[template][list]", TestIndexAccessTypes )
{
    TestType srcFrameBuffer( 4, 5, 1.0f );
    TestType dstFrameBuffer( srcFrameBuffer );
    CHECK( dstFrameBuffer.GetSize() == 20 );
    CHECK( dstFrameBuffer.GetBuffer() != nullptr );
    CHECK( dstFrameBuffer.GetBuffer() != srcFrameBuffer.GetBuffer() );
    for ( size_t i = 0; i < 20; ++i )
    {
        CHECK( srcFrameBuffer[ i ] == 1.0f );
        CHECK( dstFrameBuffer[ i ] == 1.0f );
    }
}

TEMPLATE_LIST_TEST_CASE( "FrameBuffer_GetSize", "[template][list]", TestUnifiedTypes )
{
    TestType frameBuffer;
    CHECK( frameBuffer.GetSize() == 0 );
    TestType frameBufferB( 2, 3 );
    CHECK( frameBufferB.GetSize() == 6 );
}

TEMPLATE_LIST_TEST_CASE( "FrameBuffer_IsEmpty", "[template][list]", TestUnifiedTypes )
{
    TestType frameBuffer;
    CHECK( frameBuffer.IsEmpty() );

    TestType frameBufferB( 2, 3 );
    CHECK( !frameBufferB.IsEmpty() );
}

TEMPLATE_LIST_TEST_CASE( "FrameBuffer_Resize", "[template][list]", TestUnifiedTypes )
{
    TestType frameBuffer;
    CHECK( frameBuffer.Resize( 3, 4 ) );
    CHECK( frameBuffer.GetSize() == 12 );
    CHECK( frameBuffer.GetBuffer() != nullptr );
}

TEMPLATE_LIST_TEST_CASE( "FrameBuffer_ResizeValue", "[template][list]", TestIndexAccessTypes )
{
    TestType frameBuffer;
    frameBuffer.Resize( 4, 5, 2.0f );
    CHECK( frameBuffer.GetSize() == 20 );
    CHECK( frameBuffer.GetBuffer() != nullptr );
    for ( size_t i = 0; i < 20; ++i )
    {
        CHECK( frameBuffer[ i ] == 2.0f );
    }
}

TEMPLATE_LIST_TEST_CASE( "FrameBuffer_FrameBufferElementAccess", "[template][list]", TestIndexAccessTypes )
{
    TestType frameBuffer;
    frameBuffer.Resize( 2, 2, 2.0f );
    for ( size_t i = 0; i < 4; ++i )
    {
        CHECK( frameBuffer[ i ] == 2.0f );
    }

    frameBuffer( 1, 0 ) = 3.0f;
    frameBuffer( 1, 1 ) = 4.0f;

    CHECK( frameBuffer( 0, 0 ) == 2.0f );
    CHECK( frameBuffer( 0, 1 ) == 2.0f );
    CHECK( frameBuffer( 1, 0 ) == 3.0f );
    CHECK( frameBuffer( 1, 1 ) == 4.0f );
}

TEMPLATE_LIST_TEST_CASE( "FrameBuffer_Clear", "[template][list]", TestUnifiedTypes )
{
    TestType frameBuffer( 4, 5 );
    CHECK( frameBuffer.GetSize() == 20 );
    CHECK( frameBuffer.GetBuffer() != nullptr );

    frameBuffer.Clear();
    CHECK( frameBuffer.GetSize() == 0 );
    CHECK( frameBuffer.GetBuffer() == nullptr );
}
