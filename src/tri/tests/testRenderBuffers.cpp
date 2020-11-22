#include <catch2/catch.hpp>

#include <tri/tri.h>

TEST_CASE("TriRenderBuffers")
{
    TriRenderBuffers buffers;
    REQUIRE(buffers.color.id == TriId_Uninitialized);
}

TEST_CASE("TriRenderBuffers_Create_Destroy")
{
    for (int deviceInt = TriDevice_CPU; deviceInt < TriDevice_Count;
         ++deviceInt) {
        TriDevice device = (TriDevice)deviceInt;

        TriContext ctx;
        CHECK(TriContextCreate(ctx, device) == TriStatus_Success);

        // Test creation.
        TriRenderBuffers buffers;
        REQUIRE(TriRenderBuffersCreate(buffers, ctx, 640, 480) ==
                TriStatus_Success);
        REQUIRE(buffers.color.id != TriId_Uninitialized);

        // Test destruction.
        REQUIRE(TriRenderBuffersDestroy(buffers) == TriStatus_Success);
        REQUIRE(buffers.color.id == TriId_Uninitialized);

        CHECK(TriContextDestroy(ctx) == TriStatus_Success);
    }
}
