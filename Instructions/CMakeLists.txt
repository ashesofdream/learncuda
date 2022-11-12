project(Instructions)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_LINK_LIBRARIES_KEYWORD  PRIVATE)
file(CREATE_LINK ${CMAKE_SOURCE_DIR}/resource ${CMAKE_BINARY_DIR}/resource SYMBOLIC)
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${CMAKE_CURRENT_BINARY_DIR}/shaders SYMBOLIC)

cuda_add_executable(Instructions main.cu ../include/util.cpp ../include/util.h ../include/gl_util.cpp ../include/Mesh.cpp
../include/Shader.cpp ) 
target_link_libraries(Instructions PRIVATE glm::glm glad::glad glfw)
target_include_directories(Instructions PRIVATE ${STB_INCLUDE_DIRS})