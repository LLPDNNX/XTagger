project(xtagger)
cmake_minimum_required(VERSION 3.15)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

macro(copy_file_if_changed in_file out_file target)  
    add_custom_command (
	    TARGET     ${target}
	    POST_BUILD
	    COMMAND    ${CMAKE_COMMAND}
	    ARGS       -E copy_if_different ${in_file} ${out_file}
	    COMMENT "Copying file: ${in_file} to: ${out_file}"
    )
endmacro(copy_file_if_changed)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specific - will use 'Release' as default")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release
        RelWithDebInfo MinSizeRel."
        FORCE
    )
endif(NOT CMAKE_BUILD_TYPE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "\
        -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic \
        -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual \
        -Wconversion -Wsign-conversion -Wmisleading-indentation \
        -Wduplicated-cond -Wlogical-op \
        -Wnull-dereference -Wuseless-cast -Wdouble-promotion \
        ${CMAKE_CXX_FLAGS}"
    )
endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

find_package(PythonInterp)
find_package(TensorFlow)
find_package(ROOT)

set(MODULENAME "xtagger" CACHE STRING "Module name")
message(STATUS "Module name: ${MODULENAME}")

add_definitions(${ROOT_DEFINITIONS} ${TF_DEFINITIONS})
include_directories(SYSTEM ${TensorFlow_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR})

add_library(RootReader
    MODULE
    RootReader.cc
)
target_link_libraries(RootReader ${TensorFlow_LIBRARY} ${ROOT_LIBRARIES} -lTreePlayer)
install(TARGETS RootReader LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/xtagger")

add_library(ClassificationWeights
    MODULE
    ClassificationWeights.cc
)
target_link_libraries(ClassificationWeights ${TensorFlow_LIBRARY} ${ROOT_LIBRARIES})
install(TARGETS ClassificationWeights LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/xtagger")

add_library(FakeBackground
    MODULE
    FakeBackground.cc
)
target_link_libraries(FakeBackground ${TensorFlow_LIBRARY} ${ROOT_LIBRARIES})
install(TARGETS FakeBackground LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/xtagger")

add_library(Resampler
    MODULE
    Resampler.cc
)
target_link_libraries(Resampler ${TensorFlow_LIBRARY})
install(TARGETS Resampler LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/xtagger")


set(pythonfiles
    __init__.py
    root_reader.py
    resampler.py
    fake_background.py
    classification_weights.py
)

foreach(pyfile ${pythonfiles})
    copy_file_if_changed(
        ${CMAKE_SOURCE_DIR}/${pyfile}
        ${CMAKE_CURRENT_BINARY_DIR}/${pyfile}
        RootReader
    )
    install(FILES ${CMAKE_SOURCE_DIR}/${pyfile} DESTINATION "${CMAKE_INSTALL_PREFIX}/xtagger")
endforeach(pyfile)

enable_testing()

add_test(
    NAME test_xtagger
    COMMAND ${PYTHON_EXECUTABLE} -c "import xtagger"
    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
)

