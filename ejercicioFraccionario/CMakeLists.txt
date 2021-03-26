

add_executable(matvec matvec.c)

target_link_libraries(matvec ${OPENCL_LIBRARIES})


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/matvec.cl ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

