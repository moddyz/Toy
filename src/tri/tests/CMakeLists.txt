file(GLOB CPPFILES *.cpp)
cpp_test(test_tri
    CPPFILES
        ${CPPFILES}
    LIBRARIES
        tri
        CUDA::cudart
)
