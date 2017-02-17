cmake_minimum_required(VERSION 3.6)
project(FTCS_3D)

include_directories(include)
add_executable(main main.c lib/matrix.c)