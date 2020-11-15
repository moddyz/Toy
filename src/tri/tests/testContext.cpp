#include <catch2/catch.hpp>

#include <tri/tri.h>

TEST_CASE("TriContext")
{
    TriContext ctx;
    REQUIRE(ctx.id == TriId_Uninitialized);
}

TEST_CASE("TriContextCreatePreferred")
{
    TriContext ctx;
    REQUIRE(TriContextCreatePreferred(ctx) == TriStatus_Success);
    REQUIRE(ctx.id != TriId_Uninitialized);
    CHECK(TriContextDestroy(ctx) == TriStatus_Success);
}

TEST_CASE("TriContextCreate")
{
    TriContext ctx;
    REQUIRE(TriContextCreate(ctx, TriDevice_CPU) == TriStatus_Success);
    REQUIRE(ctx.id != TriId_Uninitialized);
    CHECK(TriContextDestroy(ctx) == TriStatus_Success);

    REQUIRE(TriContextCreate(ctx, TriDevice_CUDA) == TriStatus_Success);
    REQUIRE(ctx.id != TriId_Uninitialized);
    CHECK(TriContextDestroy(ctx) == TriStatus_Success);
}

TEST_CASE("TriContextDestroy")
{
    TriContext ctx;
    CHECK(TriContextCreate(ctx, TriDevice_CPU) == TriStatus_Success);
    REQUIRE(TriContextDestroy(ctx) == TriStatus_Success);
    REQUIRE(ctx.id == TriId_Uninitialized);
}

TEST_CASE("TriContextGetDevice")
{
    TriContext ctx;
    CHECK(TriContextCreate(ctx, TriDevice_CPU) == TriStatus_Success);

    TriDevice device;
    REQUIRE(TriContextGetDevice(ctx, device) == TriStatus_Success);
    REQUIRE(device == TriDevice_CPU);
}
