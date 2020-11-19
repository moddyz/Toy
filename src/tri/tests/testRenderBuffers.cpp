#include <catch2/catch.hpp>

#include <tri/tri.h>

TEST_CASE("TriRenderBuffersCreate")
{
    for (int device = TriDevice_CPU; device < TriDevice_Count; ++device) {
        TriContext ctx;
        CHECK(TriContextCreate(ctx, (TriDevice)device) == TriStatus_Success);

        TriRenderBuffers buffers;
        REQUIRE(TriRenderBuffersCreate(buffers, ctx, 640, 480) ==
                TriStatus_Success);

        CHECK(TriRenderBuffersDestroy(buffers) == TriStatus_Success);
        CHECK(TriContextDestroy(ctx) == TriStatus_Success);
    }
}
