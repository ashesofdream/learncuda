#include "Mesh.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include "Shader.h"
using namespace std;

Mesh::Mesh(){
    
}

Mesh::Mesh(std::vector<Vertex>&& _vertices){
    vertices = _vertices;
};

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<int> indices, std::vector<Texture> textures){
    this->vertices = vertices;
    this->triangle_indices = indices;
    this->textures = textures;
    setup_triangle_mesh();
};
Mesh::Mesh(std::vector<Vertex> vertices, std::vector<int> indices)
{
    this->vertices = vertices;
    this->triangle_indices = indices;
    setup_triangle_mesh();
}
void Mesh::add_triangle_indices(const std::array<int, 3> &indices){
    triangle_indices.insert(triangle_indices.end(),indices.begin(),indices.end());    
}
void Mesh::add_quad_indices(const std::array<int, 4> &indice){
    quad_indices.insert(quad_indices.end(),indice.begin(),indice.end());
}


void Mesh::setup_triangle_mesh() {
    
    glGenVertexArrays(1,&VAO);
    glGenBuffers(1,&VBO);
    glGenBuffers(1,&EBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER,VBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(Vertex)*vertices.size(),&vertices[0],GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(unsigned int)*triangle_indices.size(),&triangle_indices[0],GL_STATIC_DRAW);

    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(struct Vertex, normal));
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*) offsetof(struct Vertex,tex_coord));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

}

void Mesh::setup_quad_mesh(){
    glGenVertexArrays(1,&quad_vao);
    glGenBuffers(1,&quad_vbo);
    glGenBuffers(1,&quad_ebo);
    glBindVertexArray(quad_vao);

    glBindBuffer(GL_ARRAY_BUFFER,quad_vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(Vertex)*vertices.size(),&vertices[0],GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,quad_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(unsigned int)*triangle_indices.size(),&triangle_indices[0],GL_STATIC_DRAW);

    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(Vertex, normal));
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*) offsetof(Vertex,tex_coord));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,0);
}

void Mesh::draw(const Shader& shader){
    std::array<vector<int>*,2> indices_array{&triangle_indices,&quad_indices};
    for(const auto & p_cur_indices:indices_array){
        auto& cur_indices = *p_cur_indices;
        if(!cur_indices.empty()){
            shader.use();
            unsigned int diffuseNr = 1,specularNr = 1;
            for(unsigned int i = 0 ; i < textures.size(); ++i){
                glActiveTexture(GL_TEXTURE0+i);
                std::string number;
                std::string name= textures[i].type;
                if(name== "texture_diffuse"){
                    number = std::to_string(diffuseNr);
                }else if(name   == "texture_specular"){
                    number = std::to_string(specularNr);
                }
                auto sampler_name = ("material."s+name+number);
                shader.set_int(sampler_name.c_str(),i);
                glBindTexture(GL_TEXTURE_2D,textures[i].id);
            }
            glActiveTexture(GL_TEXTURE0);
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, cur_indices.size(), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    }
    
}
void Mesh::emplace_vertex(Vertex && vertex){
    vertices.emplace_back(vertex);
}

const unsigned int Mesh::get_VAO() const {
    return VAO;
}

void Mesh::draw(const Shader &shader, int instance_num) {
    shader.use();
    unsigned int diffuseNr = 1,specularNr = 1;
    for(unsigned int i = 0 ; i < textures.size(); ++i){
        glActiveTexture(GL_TEXTURE0+i);
        std::string number;
        std::string name= textures[i].type;
        if(name== "texture_diffuse"){
            number = std::to_string(diffuseNr);
        }else if(name == "texture_specular"){
            number = std::to_string(specularNr);
        }
        auto sampler_name = ("material."s+name+number);
        shader.set_int(sampler_name.c_str(),i);
        glBindTexture(GL_TEXTURE_2D,textures[i].id);
    }
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(VAO);
    glDrawElementsInstanced(GL_TRIANGLES, triangle_indices.size(), GL_UNSIGNED_INT, 0,instance_num);
    glBindVertexArray(0);
}