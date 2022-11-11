project(Instructions)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_LINK_LIBRARIES_KEYWORD  PRIVATE)
cuda_add_executable(Instructions main.cu ../include/util.cpp ../include/util.h ../include/gl_util.cpp)
target_link_libraries(Instructions PRIVATE glm::glm glad::glad)
target_include_directories(Instructions PRIVATE ${STB_INCLUDE_DIRS})