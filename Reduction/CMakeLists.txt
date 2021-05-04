#deps
#1-OpenCL
include_directories(${OpenCL_INCLUDE_DIRS})

#como vamos a usar la librería en varios ejecutables y no hay problemas de nombres, se puede poner así.
include_directories(${clconv_INCLUDE_DIRS})


add_executable(reduction_complete reduction_complete.cpp)
target_link_libraries(reduction_complete clconv)
message(STATUS "exe: " ${clconv_LIBARIES})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reduction_complete.cl  ${CMAKE_CURRENT_BINARY_DIR}/../Reduction COPYONLY)


#target_include_directories(matvec PUBLIC
#                              "${PROJECT_BINARY_DIR}"
#                              ${clconv_INCLUDE_DIRECTORIES}
#                              )

