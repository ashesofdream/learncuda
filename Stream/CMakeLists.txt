project(Stream)

set(CMAKE_CXX_STANDARD 17)
cuda_add_executable(Stream main.cu ../include/util.cpp ../include/util.h OPTIONS "-std=c++17")
