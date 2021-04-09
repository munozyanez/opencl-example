#deps
#1-OpenCL
include(FindOpenCL)

include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(matvec matvec.c)
target_link_libraries(matvec OpenCL::OpenCL lib)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/matvec.cl ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

target_include_directories(matvec PUBLIC
                              "${PROJECT_BINARY_DIR}"
                              "${PROJECT_SOURCE_DIR}/lib"
                              )
