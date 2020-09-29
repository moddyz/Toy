get_filename_component(PROGRAM_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
cpp_program(
    ${PROGRAM_NAME}
    CPPFILES
        main.cpp
    LIBRARIES
        ${GLFW_LIBRARIES}
        CUDA::cudart
        toy_utils
        toy_memory
        toy_imaging
        toy_application
        gm
)
