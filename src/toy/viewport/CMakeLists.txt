file(GLOB HEADERS *.h)
file(GLOB CPPFILES *.cpp)
get_filename_component(LIBRARY_SUFFIX ${CMAKE_CURRENT_SOURCE_DIR} NAME)

cpp_library(${ROOT_PREFIX}_${LIBRARY_SUFFIX}
    TYPE
        STATIC
    HEADERS_INSTALL_PREFIX
        ${ROOT_PREFIX}/${LIBRARY_SUFFIX}
    PUBLIC_HEADERS
        ${HEADERS}
    CPPFILES
        ${CPPFILES}
    LIBRARIES
        ${GLEW_LIBRARIES}
        ${OPENGL_LIBRARIES}
        CUDA::cudart
        gm
        toy_memory
        toy_base
)

add_subdirectories()
