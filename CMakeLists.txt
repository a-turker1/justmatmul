cmake_minimum_required(VERSION 3.5.0)
project(justmatmul VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(justmatmul main.cpp src/matmul)
target_include_directories(justmatmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
