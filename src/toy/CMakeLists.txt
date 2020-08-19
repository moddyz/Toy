set(ROOT_PREFIX "pbr")

# Top level headers.
file(GLOB HEADERS *.h)
file(
    COPY ${HEADERS}
    DESTINATION ${CMAKE_BINARY_DIR}/include/${ROOT_PREFIX}
)

install(
    FILES ${HEADERS}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${ROOT_PREFIX}
)

add_subdirectories()
