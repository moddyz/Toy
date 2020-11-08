file(GLOB CPPFILES *.cpp)

cpp_library(tri
    TYPE
        STATIC
    PUBLIC_HEADERS
        tri.h
    CPPFILES
        ${CPPFILES}
    LIBRARIES
        CUDA::cudart
)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
