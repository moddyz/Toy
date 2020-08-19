file(GLOB CPPFILES *.cpp)
cpp_test_program(test_${ROOT_PREFIX}_${LIBRARY_SUFFIX}
    CPPFILES
        ${CPPFILES}
    LIBRARIES
        gm
        ${ROOT_PREFIX}_${LIBRARY_SUFFIX}
        ${ROOT_PREFIX}_utils
)
