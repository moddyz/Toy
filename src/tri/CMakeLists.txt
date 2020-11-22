file(GLOB CPPFILES *.cpp internal/*.cpp)

cpp_library(tri
    TYPE
        STATIC
    PUBLIC_HEADERS
        tri.h
    CPPFILES
        ${CPPFILES}
    LIBRARIES
        CUDA::cudart
        gm
)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
