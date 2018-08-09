project(RootReader)
cmake_minimum_required(VERSION 2.8)
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

find_package(TensorFlow)
find_package(ROOT)
find_package(PythonInterp)

set(MODULENAME "xtagger" CACHE STRING "Module name")
message(STATUS "Module name: ${MODULENAME}")

set(SITEPACKDIR ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/${MODULENAME})
message(STATUS "Install dir: ${SITEPACKDIR}")

add_definitions(${ROOT_DEFINITIONS})
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11)

include_directories(${TensorFlow_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR})
add_library(RootReader
    MODULE
    RootReader.cc
)
target_link_libraries(RootReader ${TensorFlow_LIBRARY} ${ROOT_LIBRARIES} -lTreePlayer)
install(TARGETS RootReader LIBRARY DESTINATION ${SITEPACKDIR})

add_library(RootWriter
    MODULE
    RootWriter.cc
)
target_link_libraries(RootWriter ${TensorFlow_LIBRARY} ${ROOT_LIBRARIES})
install(TARGETS RootWriter LIBRARY DESTINATION ${SITEPACKDIR})

add_library(ClassificationWeights
    MODULE
    ClassificationWeights.cc
)
target_link_libraries(ClassificationWeights ${TensorFlow_LIBRARY} ${ROOT_LIBRARIES})
install(TARGETS ClassificationWeights LIBRARY DESTINATION ${SITEPACKDIR})

add_library(FakeBackground
    MODULE
    FakeBackground.cc
)
target_link_libraries(FakeBackground ${TensorFlow_LIBRARY} ${ROOT_LIBRARIES})
install(TARGETS FakeBackground LIBRARY DESTINATION ${SITEPACKDIR})

add_library(Resampler
    MODULE
    Resampler.cc
)
target_link_libraries(Resampler ${TensorFlow_LIBRARY})
install(TARGETS Resampler LIBRARY DESTINATION ${SITEPACKDIR})


set(pythonfiles
    __init__.py
    root_reader.py
    root_writer.py
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
    install(FILES ${CMAKE_SOURCE_DIR}/${pyfile} DESTINATION ${SITEPACKDIR})
endforeach(pyfile)
