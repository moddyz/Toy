file(GLOB CPPFILES *.cpp)
cpp_test(test_${ROOT_PREFIX}_${LIBRARY_SUFFIX}
    CPPFILES
        ${CPPFILES}
    LIBRARIES
        gm
        ${ROOT_PREFIX}_${LIBRARY_SUFFIX}
        ${ROOT_PREFIX}_base
)
