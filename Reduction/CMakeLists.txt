
project(Reduction)

set(CMAKE_CXX_STANDARD 17)

cuda_add_executable(Reduction main.cu ../include/util.cpp ../include/util.h)
