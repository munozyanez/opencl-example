project(opencl-example)

cmake_minimum_required(VERSION 2.9)

#deps
#1-OpenCL
include(FindOpenCL)

include_directories(${OpenCL_INCLUDE_DIRS})

add_subdirectory(ejercicioFraccionario ejercicioFraccionario)


##functions
function(echo_all_cmake_variable_values)
  message(STATUS "")
  get_cmake_property(vs VARIABLES)
  foreach(v ${vs})
    message(STATUS "${v}='${${v}}'")
  endforeach(v)
  message(STATUS "")
endfunction()


echo_all_cmake_variable_values()

