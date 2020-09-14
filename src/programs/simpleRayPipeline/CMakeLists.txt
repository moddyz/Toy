cpp_program(simpleRayPipelineHost
    CPPFILES
        simpleRayPipelineHost.cpp
    LIBRARIES
        ${GLFW_LIBRARIES}
        CUDA::cudart
        toy_utils
        toy_memory
        toy_imaging
        toy_application
        gm
)

#   cpp_program(simpleRayPipelineCuda
#       CPPFILES
#           simpleRayPipelineCuda.cpp
#       LIBRARIES
#           ${GLFW_LIBRARIES}
#           CUDA::cudart
#           toy_utils
#           toy_memory
#           toy_imaging
#           toy_application
#           gm
#   )
