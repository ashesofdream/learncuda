#include<vector>
#include <glm/glm.hpp>
#include <string>
#include <optional>
#include <array>
class Shader;
struct Vertex{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 tex_coord;
};
struct Texture{
    unsigned int id;
    std::string type;
    const char* path;
};

class Mesh {
private:
    std::vector<Vertex> vertices;
    std::vector<int> quad_indices;
    std::vector<int> triangle_indices;
    std::vector<Texture> textures;
    int indices_size;
    unsigned int VBO,VAO,EBO;
    unsigned int quad_vbo,quad_vao,quad_ebo;
public:
    Mesh();
    Mesh(std::vector<Vertex>&& _vertices);
    Mesh(std::vector<Vertex> vertices, std::vector<int> indices, std::vector<Texture> textures);
    Mesh(std::vector<Vertex> vertices, std::vector<int> indices);
    void draw(const Shader& shader);
    void draw(const Shader& shader,int instance_num);
    void add_triangle_indices(const std::array<int, 3> & indice);
    void add_quad_indices(const std::array<int,4>& indice);
    void emplace_vertex(Vertex && vertex);
    void setup_quad_mesh();
    void setup_triangle_mesh();
    const unsigned int get_VAO() const;
};
