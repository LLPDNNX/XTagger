macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      add_definitions ("-std=c++11")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

project(unpacker)

use_cxx11()
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

find_package(ROOT)

include_directories(${ROOT_INCLUDE_DIR} ${PROJECT_SOURCE_DIR})

add_executable(unpackNanoXFast
    unpackNanoXFast.cc
)
target_link_libraries(unpackNanoXFast ${ROOT_LIBRARIES} -lTreePlayer)
