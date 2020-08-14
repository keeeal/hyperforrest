#version 430

in vec4 vertex;
in vec4 normal;
in vec4 colour;

out VertexData
{
    vec4 vertex;
    vec4 normal;
    vec4 colour;
} vertex_data;

void main()
{
    vertex_data.vertex = vertex;
    vertex_data.normal = normal;
    vertex_data.colour = colour;
}