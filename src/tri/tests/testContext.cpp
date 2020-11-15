#include <catch2/catch.hpp>

#include <tri/tri.h>

TEST_CASE( "TriContextCreatePreferred" )
{
    TriContext ctx;
    CHECK( TriContextCreatePreferred( ctx ) == TriStatus_Success );
}
