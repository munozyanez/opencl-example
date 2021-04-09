project(FracionalControl)

cmake_minimum_required(VERSION 2.9)

#Hay que añadir la librería antes para poder compilar luego.
add_subdirectory(lib)

add_subdirectory(ejercicioFraccionario ejercicioFraccionario)

#project(FracionalControl) no se pueden poner varios proyectos en un archivo


##functions
function(echo_all_cmake_variable_values)
  message(STATUS "")
  get_cmake_property(vs VARIABLES)
  foreach(v ${vs})
    message(STATUS "${v}='${${v}}'")
  endforeach(v)
  message(STATUS "")
endfunction()


#echo_all_cmake_variable_values()

