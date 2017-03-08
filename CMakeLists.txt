cmake_minimum_required(VERSION 3.6)
project(FTCS_3D)

include_directories(include)

add_executable(main main.c lib/matrix.c lib/grid.c)
target_link_libraries(main m)

add_executable(cart_demo cart_demo.c lib/matrix.c lib/grid.c)
target_link_libraries(cart_demo m)
