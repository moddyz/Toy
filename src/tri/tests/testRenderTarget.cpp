#include <catch2/catch.hpp>

#include <tri/tri.h>

TEST_CASE("TriRenderTarget")
{
    TriRenderTarget target;
    REQUIRE(target.id == TriId_Uninitialized);
}

TEST_CASE("TriRenderTarget_Create_Destroy")
{
    for (int deviceInt = TriDevice_CPU; deviceInt < TriDevice_Count;
         ++deviceInt) {
        TriDevice device = (TriDevice)deviceInt;

        TriContext ctx;
        CHECK(TriContextCreate(ctx, device) == TriStatus_Success);

        // Test creation.
        TriRenderTarget target;
        REQUIRE(TriRenderTargetCreate(target, ctx, 640, 480) ==
                TriStatus_Success);
        REQUIRE(target.id != TriId_Uninitialized);

        // Test destruction.
        REQUIRE(TriRenderTargetDestroy(target) == TriStatus_Success);
        REQUIRE(target.id == TriId_Uninitialized);

        CHECK(TriContextDestroy(ctx) == TriStatus_Success);
    }
}

TEST_CASE("TriRenderTargetBuffer")
{
    for (int deviceInt = TriDevice_CPU; deviceInt < TriDevice_Count;
         ++deviceInt) {
        TriDevice device = (TriDevice)deviceInt;

        TriContext ctx;
        CHECK(TriContextCreate(ctx, device) == TriStatus_Success);

        TriRenderTarget target;
        CHECK(TriRenderTargetCreate(target, ctx, 640, 480) ==
              TriStatus_Success);
        CHECK(target.id != TriId_Uninitialized);

        // Query color buffer
        TriBuffer colorBuffer;
        REQUIRE(TriRenderTargetBuffer(target, "color", colorBuffer) ==
                TriStatus_Success);
        REQUIRE(colorBuffer.id != TriId_Uninitialized);

        // Query non-existent buffer.
        TriBuffer fooBuffer;
        REQUIRE(TriRenderTargetBuffer(target, "foo", fooBuffer) ==
                TriStatus_BufferNotFound);
        REQUIRE(fooBuffer.id == TriId_Uninitialized);

        CHECK(TriRenderTargetDestroy(target) == TriStatus_Success);
        CHECK(target.id == TriId_Uninitialized);

        CHECK(TriContextDestroy(ctx) == TriStatus_Success);
    }
}
