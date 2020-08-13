#version 430

layout(triangles) in;
layout(triangle_strip, max_vertices=4) out;

uniform mat4 p3d_ModelViewProjectionMatrix;

out vec4 vertex_color;

//
void main()
{
    int i;
    for (i = 0; i < 4; i ++)
    {
        vec4 projected = vec4(gl_in[i].gl_Position.xyz, 1.0)
        gl_Position = p3d_ModelViewProjectionMatrix * projected;
        vertex_color = vec4(0.0, 1.0, 0.0, 1.0);
        EmitVertex();
    }
    EndPrimitive();
}