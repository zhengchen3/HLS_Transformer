cmake_minimum_required(VERSION 3.20)
project(NN_3)

set(CMAKE_CXX_STANDARD 14)

add_executable(NN_3 main.cpp values_4_fill.hpp params.h)
add_compile_options(-bigobj)