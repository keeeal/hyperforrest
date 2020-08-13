#version 430

in vec4 p3d_Vertex;

// pass-through vertex shader
void main()
{
    gl_Position = p3d_Vertex;
}