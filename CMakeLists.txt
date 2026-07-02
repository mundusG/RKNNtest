cmake_minimum_required(VERSION 3.14)
project(test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBS_DIR ${CMAKE_SOURCE_DIR}/libs/aarch64)

include_directories(
    ${LIBS_DIR}/include
    ${LIBS_DIR}/include/rga
)

link_directories(${LIBS_DIR}/lib)

add_executable(test test.cpp)
add_executable(gen_test_img gen_test_img.cpp)

target_link_libraries(test rknn_api rknnrt rga)
target_link_libraries(gen_test_img rknn_api rknnrt rga)
