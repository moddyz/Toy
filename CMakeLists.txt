cmake_minimum_required(VERSION 3.17)

# Project Name
project(
    Toy
    VERSION 0.0
    DESCRIPTION "Building blocks for producing images."
    LANGUAGES CXX CUDA
)

# Discover macros.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/packages)

# Apply project options, defaults, and dependencies.
include(Options)
include(Defaults)
include(Packages)

# Build utilities.
include(BaseTools)
include(CXXTools)

# Build targets.
add_subdirectory(thirdparty)
add_subdirectory(src)
