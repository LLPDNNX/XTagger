macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
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

include_directories(${TensorFlow_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR} ${PROJECT_SOURCE_DIR})

add_executable(unpackNanoX
    unpackNanoX.cc
)
target_link_libraries(unpackNanoX ${ROOT_LIBRARIES} -lTreePlayer)

add_executable(unpackNanoXFast
    unpackNanoXFast.cc
)
target_link_libraries(unpackNanoXFast ${ROOT_LIBRARIES} -lTreePlayer)
