cmake_minimum_required(VERSION 3.6)
project(FTCS_3D)

include_directories(include /home/rahaman/install/mpich-3.2-gnu-5/include)

add_executable(main main.c lib/matrix.c lib/cart.c)
target_link_libraries(main m)

add_executable(cart_demo cart_demo.c lib/matrix.c lib/cart.c)
target_link_libraries(cart_demo m)
