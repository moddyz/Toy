#include <catch2/catch.hpp>

#include <tri/tri.h>

TEST_CASE("TriRenderer")
{
    TriRenderer renderer;
    REQUIRE(renderer.id == TriId_Uninitialized);
}

TEST_CASE("TriRendererCreate")
{
    for (int deviceInt = TriDevice_CPU; deviceInt < TriDevice_Count;
         ++deviceInt) {
        TriDevice device = (TriDevice)deviceInt;

        TriContext ctx;
        CHECK(TriContextCreate(ctx, device) == TriStatus_Success);

        TriRenderer renderer;
        REQUIRE(TriRendererCreate(renderer, ctx) == TriStatus_Success);
        REQUIRE(renderer.id != TriId_Uninitialized);

        CHECK(TriRendererDestroy(renderer) == TriStatus_Success);
        CHECK(TriContextDestroy(ctx) == TriStatus_Success);
    }
}
