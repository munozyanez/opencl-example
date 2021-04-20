#deps
#1-OpenCL
include_directories(${OpenCL_INCLUDE_DIRS})

#como vamos a usar la librería en varios ejecutables y no hay problemas de nombres, se puede poner así.
include_directories(${clconv_INCLUDE_DIRS})


add_executable(matvec matvec.cpp)
target_link_libraries(matvec clconv)
message(STATUS "exe: " ${clconv_LIBARIES})



#target_include_directories(matvec PUBLIC
#                              "${PROJECT_BINARY_DIR}"
#                              ${clconv_INCLUDE_DIRECTORIES}
#                              )

