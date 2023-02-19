cmake_minimum_required(VERSION 3.23)
project(YAInfer)

set(CMAKE_CXX_STANDARD 17)

# set(link_libraries glog pthread gtest)
# set(link_math_libraries armadillo blas lapack)

add_executable(YAInfer main.cpp)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# target_include_directories(YAInfer PUBLIC /usr/include/armadillo_bits)

# target_include_directories(YAInfer PUBLIC ${GTEST_INCLUDE_DIRS})

# target_include_directories(YAInfer PUBLIC ${link_libraries} ${link_math_libraries})
# target_include_directories(YAInfer PUBLIC ${link_libraries} ${link_math_libraries})
include_directories(./include)
enable_testing()
add_subdirectory(test)