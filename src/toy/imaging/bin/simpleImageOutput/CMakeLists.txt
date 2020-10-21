get_filename_component(PROGRAM_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
cpp_executable(
    ${PROGRAM_NAME}
    CPPFILES
        main.cpp
    LIBRARIES
        CUDA::cudart
        toy_base
        toy_memory
        toy_imaging
        gm
        cxxopts
)
