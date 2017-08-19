cmake_minimum_required(VERSION 2.8.12)

project(kmeans)

# Set exe path.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Set library path.`
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_package(Threads REQUIRED)

# Set C++ flags.
set(CMAKE_CXX_FLAGS 
  "${CMAKE_CXX_FLAGS} -std=c++11 -g")

# Set the module path (mainly for FindXXX.cmake files)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
											${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)



include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# Compile and link render_views executable.
add_executable(kmeans_test kmeans_test.cc kdtree.cc)
target_link_libraries (kmeans_test ${CMAKE_THREAD_LIBS_INIT})
