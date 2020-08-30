cmake_minimum_required(VERSION 3.17)

# Project Name
project(
    Toy
    VERSION 0.0
    DESCRIPTION "A CUDA-based toy ray tracer."
    LANGUAGES CXX CUDA
)

find_package(CUDAToolkit)

# Options
option(BUILD_DOCUMENTATION "Build doxygen documentation." ON)

# Cross-platform compatibility.
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# Discover macros.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
include(Public)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

# Testing.
if(BUILD_TESTING)
    enable_testing()
    list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
endif()

if(BUILD_DOCUMENTATION)
    file(
        COPY ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        DESTINATION ${CMAKE_BINARY_DIR}
    )

    doxygen_documentation(
        doxygen_docs
        GENERATE_TAGFILE
            TRUE
        DOXYFILE
            ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
        INPUTS
            ${CMAKE_CURRENT_SOURCE_DIR}/src/
    )
endif()

# Build targets.
add_subdirectory(thirdparty)
add_subdirectory(src)

# Install exported targets.
#export_project(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in)
