cmake_minimum_required(VERSION 3.14)
project(BenchMath)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# Google Benchmark
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
FetchContent_MakeAvailable(googlebenchmark)

# Eigen
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# GLM
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

# OpenCV
FetchContent_Declare(
  OpenCV
  GIT_REPOSITORY https://github.com/opencv/opencv.git
  GIT_TAG 4.9.0
)
set(BUILD_LIST "core")
set(BUILD_TESTS OFF)
set(BUILD_PERF_TESTS OFF)
set(BUILD_EXAMPLES OFF)
FetchContent_MakeAvailable(OpenCV)

# Vectorial
FetchContent_Declare(
  vectorial
  GIT_REPOSITORY https://github.com/scoopr/vectorial
  GIT_TAG master
)
FetchContent_GetProperties(vectorial)
if(NOT vectorial_POPULATED)
        FetchContent_Populate(vectorial)
endif()
add_library(vectorial INTERFACE)
target_include_directories(vectorial INTERFACE ${vectorial_SOURCE_DIR}/include)

add_executable(BenchMath main.cpp)
target_link_libraries(BenchMath benchmark::benchmark Eigen3::Eigen glm opencv_core vectorial)
target_include_directories(BenchMath PRIVATE
        ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
        ${OPENCV_MODULE_opencv_core_LOCATION}/include
        ${OPENCV_MODULE_opencv_highgui_LOCATION}/include
        )
