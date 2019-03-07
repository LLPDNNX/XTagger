project(unpacker)

cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

find_package(ROOT)

include_directories(${ROOT_INCLUDE_DIR} ${PROJECT_SOURCE_DIR})
add_definitions(${ROOT_DEFINITIONS})
add_executable(unpackNanoXFast
    unpackNanoXFast.cc
)
target_link_libraries(unpackNanoXFast ${ROOT_LIBRARIES} -lTreePlayer)
