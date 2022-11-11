#include"MeshSurface.h"
#include "Shader.h"
void MeshSurface::add_mesh(Mesh&& meshe){
    meshes.emplace_back(meshe);
}
void MeshSurface::draw(const Shader& s){
    for(auto& mesh:meshes) mesh.draw(s);
};