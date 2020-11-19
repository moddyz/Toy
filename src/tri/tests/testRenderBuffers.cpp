#include <catch2/catch.hpp>

#include <tri/tri.h>

TEST_CASE("TriRenderBuffersCreate")
{
    TriContext ctx;
    CHECK(TriContextCreatePreferred(ctx) == TriStatus_Success);

    TriRenderBuffers buffers;
    REQUIRE(TriRenderBuffersCreate(buffers, ctx, 640, 480) == TriStatus_Success);
}
