#include <vector>
#include "Mesh.h"

class Shader;

class MeshSurface{
private:
    std::vector<Mesh> meshes;
public:
    void add_mesh(Mesh&& mesh);
    void draw(const Shader& s);
};