#include <catch2/catch.hpp>

#include <tri/tri.h>

TEST_CASE("TriRenderBuffers")
{
    TriRenderBuffers buffers;
    REQUIRE(buffers.color.ptr == nullptr);
    REQUIRE(buffers.color.numElements == 0);
    REQUIRE(buffers.color.format == TriFormat_Uninitialized);
    REQUIRE(buffers.color.device == TriDevice_Uninitialized);
}

TEST_CASE("TriRenderBuffersCreate")
{
    for (int deviceInt = TriDevice_CPU; deviceInt < TriDevice_Count;
         ++deviceInt) {
        TriDevice device = (TriDevice)deviceInt;

        TriContext ctx;
        CHECK(TriContextCreate(ctx, device) == TriStatus_Success);

        TriRenderBuffers buffers;
        REQUIRE(TriRenderBuffersCreate(buffers, ctx, 640, 480) ==
                TriStatus_Success);

        REQUIRE(buffers.color.ptr != nullptr);
        REQUIRE(buffers.color.numElements == 640 * 480);
        REQUIRE(buffers.color.format == TriFormat_Float32_Vec4);
        REQUIRE(buffers.color.device == device);

        CHECK(TriRenderBuffersDestroy(buffers) == TriStatus_Success);
        CHECK(TriContextDestroy(ctx) == TriStatus_Success);
    }
}
