project(DynamicParallel)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(DynamicParallel main.cu ../include/util.cpp ../include/util.h
OPTIONS " -std=c++17")
target_link_libraries(DynamicParallel  cudadevrt)
