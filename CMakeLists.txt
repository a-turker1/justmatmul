cmake_minimum_required(VERSION 3.5.0)
project(justmatmul VERSION 0.1.0 LANGUAGES C CXX)

set (CMAKE_CXX_STANDARD 11)
include(CTest)
set(CMAKE_CXX_FLAGS "-O3 -funroll-loops -std=c++11 -fomit-frame-pointer")
enable_testing()

add_executable(justmatmul main.cpp src/matmul.cpp)
target_include_directories(justmatmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
