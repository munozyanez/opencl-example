project(clconv)

include(FindOpenCL)


aux_source_directory(. clconv_SRCS)


SET(clconv_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Library include directories" FORCE)
message(STATUS "clconv_INCLUDE_DIRS: " ${clconv_INCLUDE_DIRS})

SET(clconv_LIBARIES ${PROJECT_NAME} CACHE STRING "Library directories" FORCE)
message(STATUS "clconv_LIBARIES: " ${PROJECT_NAME})

add_library(${PROJECT_NAME} SHARED ${clconv_SRCS} ${clconv_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reduction_complete.cl  ${CMAKE_CURRENT_BINARY_DIR}/../ejercicioFraccionario COPYONLY)

