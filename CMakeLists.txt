project(opencl-example)

cmake_minimum_required(VERSION 2.9)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

#deps
#1-OpenCL
include(FindOpenCL)

message(STATUS "OpenCL found: ${OPENCL_FOUND}")
message(STATUS "OpenCL includes: ${OPENCL_INCLUDE_DIRS}")
message(STATUS "OpenCL CXX includes: ${OPENCL_HAS_CPP_BINDINGS}")
message(STATUS "OpenCL libraries: ${OPENCL_LIBRARIES}")

include_directories(${OPENCL_INCLUDE_DIRS})

add_subdirectory(ejercicioFraccionario ejercicioFraccionario)






##functions
#function(echo_all_cmake_variable_values)
#  message(STATUS "")
#  get_cmake_property(vs VARIABLES)
#  foreach(v ${vs})
#    message(STATUS "${v}='${${v}}'")
#  endforeach(v)
#  message(STATUS "")
#endfunction()

#echo_all_cmake_variable_values()
