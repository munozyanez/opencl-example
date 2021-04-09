project(FracionalControl)

cmake_minimum_required(VERSION 2.9)



add_subdirectory(ejercicioFraccionario ejercicioFraccionario)



add_subdirectory(lib)




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

