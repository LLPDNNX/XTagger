project(unpacker)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 3.1)
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
