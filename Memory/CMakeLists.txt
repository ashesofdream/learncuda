project(Memory)

set(CMAKE_CXX_STANDARD 17)

cuda_add_executable(Memory main.cu ../include/util.cpp ../include/util.h)
