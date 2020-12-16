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

TEST_CASE("TriRendererDestroy_UninitializedRenderer")
{
    // Destroy uninitialized renderer.
    TriRenderer renderer;
    CHECK(TriRendererDestroy(renderer) == TriStatus_RendererNotFound);
}

TEST_CASE("TriRendererDestroy")
{
    TriContext ctx;
    CHECK(TriContextCreate(ctx, TriDevice_CPU) == TriStatus_Success);

    TriRenderer renderer;
    CHECK(TriRendererCreate(renderer, ctx) == TriStatus_Success);

    REQUIRE(TriRendererDestroy(renderer) == TriStatus_Success);
    REQUIRE(renderer.id == TriId_Uninitialized);

    CHECK(TriContextDestroy(ctx) == TriStatus_Success);
}

TEST_CASE("TriRendererCamera")
{
    TriContext ctx;
    CHECK(TriContextCreate(ctx, TriDevice_CPU) == TriStatus_Success);

    TriRenderer renderer;
    CHECK(TriRendererCreate(renderer, ctx) == TriStatus_Success);

    float cameraXform[16];
    REQUIRE(TriRendererCamera(renderer, cameraXform) == TriStatus_Success);

    CHECK(TriRendererDestroy(renderer) == TriStatus_Success);
    CHECK(TriContextDestroy(ctx) == TriStatus_Success);
}

TEST_CASE("TriRendererPerspective")
{
    TriContext ctx;
    CHECK(TriContextCreate(ctx, TriDevice_CPU) == TriStatus_Success);

    TriRenderer renderer;
    CHECK(TriRendererCreate(renderer, ctx) == TriStatus_Success);

    REQUIRE(TriRendererPerspective(renderer, 90, 1.5, 0.1, 1000.0) ==
            TriStatus_Success);

    CHECK(TriRendererDestroy(renderer) == TriStatus_Success);
    CHECK(TriContextDestroy(ctx) == TriStatus_Success);
}

TEST_CASE("TriRendererViewport")
{
    TriContext ctx;
    CHECK(TriContextCreate(ctx, TriDevice_CPU) == TriStatus_Success);

    TriRenderer renderer;
    CHECK(TriRendererCreate(renderer, ctx) == TriStatus_Success);

    REQUIRE(TriRendererViewport(renderer, 0, 0, 640, 480) == TriStatus_Success);

    CHECK(TriRendererDestroy(renderer) == TriStatus_Success);
    CHECK(TriContextDestroy(ctx) == TriStatus_Success);
}
