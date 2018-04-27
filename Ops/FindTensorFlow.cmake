include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

execute_process(
    COMMAND python -c "import tensorflow as tf; print 'tf_version',tf.__version__"
    OUTPUT_VARIABLE TF_VER
    ERROR_VARIABLE TF_VER
    RESULT_VARIABLE TF_VER_OK
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "tf_version (.*)" _ ${TF_VER})
set(TF_VER ${CMAKE_MATCH_1})


string(REPLACE "." ";" VERSION_LIST ${TF_VER})
list(GET VERSION_LIST 0 TF_VER_MAJOR)
list(GET VERSION_LIST 1 TF_VER_MINOR)
list(GET VERSION_LIST 2 TF_VER_PATCH)

message(STATUS "Tensorflow version: " ${TF_VER_MAJOR}.${TF_VER_MINOR}.${TF_VER_PATCH})

execute_process(
    COMMAND python -c "import tensorflow as tf; print 'tf_includepath',tf.sysconfig.get_include()"
    OUTPUT_VARIABLE TF_INC
    ERROR_VARIABLE TF_INC
    RESULT_VARIABLE TF_INC_OK
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "tf_includepath (.*)" _ ${TF_INC})
set(TF_INC ${CMAKE_MATCH_1})


if (${TF_INC_OK} EQUAL 0)
    find_path(TensorFlow_INCLUDE_DIR
        NAMES tensorflow/core/framework/op.h
        PATHS ${TF_INC}
        NO_DEFAULT_PATH
    )
endif (${TF_INC_OK} EQUAL 0)
message(STATUS ${TF_INC})

if (TF_VER_MINOR GREATER 3)

    execute_process(
        COMMAND python -c "import tensorflow as tf; print 'tf_libpath',tf.sysconfig.get_lib()"
        OUTPUT_VARIABLE TF_LIB
        ERROR_VARIABLE TF_LIB
        RESULT_VARIABLE TF_LIB_OK
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "tf_libpath (.*)" _ ${TF_LIB})
    set(TF_LIB ${CMAKE_MATCH_1})

    message(STATUS ${TF_LIB})
    if (${TF_LIB_OK} EQUAL 0)
        find_library(TensorFlow_LIBRARY 
                NAMES tensorflow_framework
                PATHS ${TF_LIB}
                NO_DEFAULT_PATH
        )
    endif (${TF_LIB_OK} EQUAL 0)
    find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)
    
else (TF_VER_MINOR GREATER 3)
    message(STATUS "Lib not required for versions 1.3.0<=")
    set(TensorFlow_LIBRARY "")
    find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR)
endif (TF_VER_MINOR GREATER 3)

# set TensorFlow_FOUND
#find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

# set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR} ${TensorFlow_INCLUDE_DIR}/external/nsync/public) #fix: https://github.com/sadeepj/crfasrnn_keras/issues/19
endif()

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)
