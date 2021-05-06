#deps
#1-OpenCL
include_directories(${OpenCL_INCLUDE_DIRS})

#como vamos a usar la librería en varios ejecutables y no hay problemas de nombres, se puede poner así.
include_directories(${clconv_INCLUDE_DIRS})


add_executable(wg_test wg_test.cpp)
target_link_libraries(wg_test clconv)
message(STATUS "exe: " ${clconv_LIBARIES})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blank.cl  ${CMAKE_CURRENT_BINARY_DIR}/../wg_test COPYONLY)


#target_include_directories(matvec PUBLIC
#                              "${PROJECT_BINARY_DIR}"
#                              ${clconv_INCLUDE_DIRECTORIES}
#                              )

