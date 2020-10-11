# Discover project dependencies.

find_package(GLFW REQUIRED)

# GLEW Workaround.
find_package(GLEW REQUIRED)
if(NOT GLEW_LIBRARIES)
    set(GLEW_LIBRARIES GLEW::GLEW)
endif()

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED 10.0)
