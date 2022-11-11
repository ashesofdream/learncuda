#include "gl_util.h"
#include "Shader.h"
#include "Mesh.h"
#include"util.h"
#include"gl_util.h"
#include<iostream>
#include<cstdio>
#include<vector>
#include <omp.h>
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include <GL/glu.h>
using namespace std;
#include <cuda_gl_interop.h>

#define GRAVATITY_CONSTANT 6.67e-11
//three body problem 

struct Body
{
    glm::vec3 velocity;
    glm::vec3 pos;
    float m;
};

__global__ void update_bodys(Body* bodys,glm::mat4 * mat_array,int bodys_num,float delta_t){
    __shared__ float distance_square[32][32];
    static glm::mat4 e_matrix(1.f);
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= bodys_num) return;
    __syncthreads();
    glm::vec3 pos = bodys[idx].pos;
    int near_idx = (idx+1)%bodys_num;
    glm::vec3 a = {0.f,0.f,0.f};
    #pragma unroll
    for(int i = 0 ; i < bodys_num ; ++i){
        glm::vec3 dis =  bodys[i].pos-pos;
        // distance_square[idx][i] =  dis.x * dis.x + dis.y *dis.y + dis.z * dis.z;
        float dis2 = dis.x * dis.x + dis.y *dis.y + dis.z * dis.z;
        a +=  glm::vec3(GRAVATITY_CONSTANT * bodys[i].m  /dis2)*normalize(dis) ;
    }
    glm::vec3 velocity(bodys[idx].velocity);
    velocity += a;
    bodys[idx].velocity = velocity;
    bodys[idx].pos = glm::vec3(delta_t) *velocity + pos;
    mat_array[idx] = glm::translate(e_matrix,bodys->pos);
}


int main(){
    glm::mat4 e_matrix(1.f);
    float delta_t = 0.1;
    float weight = 6e10;
    float g = 9.8;
    int bodys_num = 3;
    vector<Body> bodys(3,{glm::vec3(0.f),glm::vec3(0.f),weight});
    Body* body_dev_pointers = nullptr;
    vector<cudaStream_t> streams(3,nullptr);
    vector<glm::vec3> pos(bodys_num*3,glm::vec3(0.f));
    
    auto window = util::prepare_window();
    // unsigned int pos_vbo;
    // glGenBuffers(1,&pos_vbo);
    // glBindBuffer(GL_ARRAY_BUFFER,pos_vbo);
    // glBufferData(GL_ARRAY_BUFFER,sizeof(glm::vec3)*bodys_num,nullptr,GL_DYNAMIC_DRAW);
    // glBindBuffer(GL_ARRAY_BUFFER,0);
    unsigned int model_uniform_buffer;
    glGenBuffers(1,&model_uniform_buffer);
    glBindBuffer(GL_UNIFORM_BUFFER,model_uniform_buffer);
    glBufferData(GL_UNIFORM_BUFFER,sizeof(glm::mat4)*bodys_num,nullptr,GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER,0);
    
    cudaGraphicsResource_t  model_uniform_cuda_resource;
    cudaGraphicsGLRegisterBuffer(&model_uniform_cuda_resource,model_uniform_buffer,cudaGraphicsMapFlagsWriteDiscard);
    
    for(auto& stream :streams){
        cudaStreamCreate(&stream);
    }
    //TODO: finish shader
    Shader shader("","");
    cudaMalloc(&body_dev_pointers,sizeof(Body)*bodys_num);
    auto sphere_model  = util::draw_sphere(32,32);
    while(!glfwWindowShouldClose(window)){
        cudaGraphicsMapResources(1,&model_uniform_cuda_resource,0);
        void * cuda_model_matrix_array;
        size_t cuda_model_matrix_array_size;
        cudaGraphicsResourceGetMappedPointer(&cuda_model_matrix_array,&cuda_model_matrix_array_size,model_uniform_cuda_resource);
        update_bodys<<<1,bodys_num>>>(body_dev_pointers,static_cast<glm::mat4*>(cuda_model_matrix_array),bodys_num,delta_t);
        cudaGraphicsUnmapResources(1,&model_uniform_cuda_resource,0);

        //begin draw
        shader.use();
        for(int i = 0 ; i < bodys_num ; ++i ){
            sphere_model.draw(shader);
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();
        glClearColor(0.f,0.f,0.f,0.f);
        
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    }    
}