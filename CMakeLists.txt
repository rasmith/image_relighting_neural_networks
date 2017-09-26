cmake_minimum_required(VERSION 2.8.12)

project(kmeans)

# Set exe path.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Set library path.`
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_package(Threads REQUIRED)

# Set C++ flags.
#set(CMAKE_CXX_FLAGS
  #"${CMAKE_CXX_FLAGS} -std=c++11 -g -fsanitize=address\
  #-fno-omit-frame-pointer -Wno-macro-redefined")
set(CMAKE_CXX_FLAGS 
  "${CMAKE_CXX_FLAGS} -std=c++11 -O3 ")
#set(CMAKE_CXX_FLAGS 
  #"${CMAKE_CXX_FLAGS} -std=c++11 -g ")

# Set the module path (mainly for FindXXX.cmake files)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
											${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

#set(PYTHON_LIBRARY "/usr/lib64/libpython3.6m.so")
#set(PYTHON_INCLUDE_DIR "/usr/include/python3.6m")
set(PYTHON_LIBRARY "/Users/randallsmith/anaconda/lib/libpython3.6m.dylib")
set(PYTHON_INCLUDE_DIR "/Users/randallsmith/anaconda/include/python3.6m")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/glm")

# Compile and link kmeans_test executable.
add_executable(kmeans_test kmeans_test.cc  kmeans2d.cc kdtree.cc image.cc)
target_link_libraries (kmeans_test ${CMAKE_THREAD_LIBS_INIT})

# Compile and link kdtree_test executable.
add_executable(kdtree_test kdtree_test.cc kdtree.cc)
target_link_libraries (kdtree_test ${CMAKE_THREAD_LIBS_INIT})

# Compile and link quickselect_test executable.
add_executable(quickselect_test quickselect_test.cc)
target_link_libraries (quickselect_test ${CMAKE_THREAD_LIBS_INIT})

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

set_source_files_properties(kmeans2d.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(kmeans2d.i PROPERTIES SWIG_FLAGS 
                            "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
swig_add_module(kmeans2d python kmeans2d.i kmeans2d.cc kdtree.cc)
swig_link_libraries(kmeans2d ${PYTHON_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

