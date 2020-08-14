#version 430

in vec4 vertex_colour;
in vec4 vertex_normal;

out vec4 colour;

void main()
{
    colour = vertex_colour;
}