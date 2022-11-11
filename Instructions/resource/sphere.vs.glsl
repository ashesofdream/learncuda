#version 430 core
layout(location = 0) in vec3 apos;
layout(location = 1) in vec3 anormal;

layout(std140,binding = 0) uniform pos_model_matrix{
    mat4 pos_matrix[3];
};

uniform int idx;
uniform mat4 projection;
uniform mat4 view;

out vec3 pos;
out vec3 normal;

void main(){
    pos = vec3(projection * view * pos_matrix[idx] * vec4(pos,1.f));
    normal = normal;
}