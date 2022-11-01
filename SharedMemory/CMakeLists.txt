project(ShadredMemory)

set(CMAKE_CXX_STANDARD 17)

cuda_add_executable(ShadredMemory main.cu ../include/util.cpp ../include/util.h)
