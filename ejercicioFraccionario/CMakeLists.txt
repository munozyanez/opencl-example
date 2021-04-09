#deps
#1-OpenCL
include(FindOpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})

#como vamos a usar la librería en varios ejecutables y no hay problemas de nombres, se puede poner así.
include_directories(${clconv_INCLUDE_DIRS})


add_executable(matvec matvec.cpp)
target_link_libraries(matvec OpenCL::OpenCL clconv)
message(STATUS "exe: " ${clconv_LIBARIES})


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/matvec.cl ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

#target_include_directories(matvec PUBLIC
#                              "${PROJECT_BINARY_DIR}"
#                              ${clconv_INCLUDE_DIRECTORIES}
#                              )
