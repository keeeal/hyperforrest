#version 430

layout(lines_adjacency) in;
layout(triangle_strip, max_vertices=5) out;

uniform mat4 p3d_ModelViewProjectionMatrix;

uniform vec4 plane_origin;
uniform vec4 plane_normal;
uniform mat4 plane_basis;

in VertexData
{
    vec4 vertex;
    vec4 normal;
    vec4 color;
} vertex_data[];

out vec4 vertex_color;

void main()
{
    vec4 v_dot_n;
    for (int i = 0; i < 4; i ++)
    {
        v_dot_n[i] = dot(vertex_data[i].vertex, plane_normal);
    }
    float q_dot_n = dot(plane_origin, plane_normal);

    // view plane culling
    vec4 _q = {q_dot_n, q_dot_n, q_dot_n, q_dot_n};
    if (all(lessThan(v_dot_n, _q))) return;
    if (all(greaterThan(v_dot_n, _q))) return;

    // a and b index each pair of points in the tetrahedron
    int a[6] = {0, 0, 1, 1, 2, 3};
    int b[6] = {1, 2, 2, 3, 3, 0};

    vec4 first_v;
    vec4 first_n;
    vec4 first_c;
    bool first = true;

    int n = 0;
    float f[6];
    for (int i = 0; i < 6; i ++)
    {
        vec4 v_a = vertex_data[a[i]].vertex;
        vec4 v_b = vertex_data[b[i]].vertex;
        // vec4 n_a = vertex_data[a[i]].normal;
        // vec4 n_b = vertex_data[b[i]].normal;
        vec4 c_a = vertex_data[a[i]].color;
        vec4 c_b = vertex_data[b[i]].color;

        // calculate f, the fraction of the line connecting a and b that lies
        // to a's side of the plane
        float v_a_dot_n = dot(v_a, plane_normal);
        float v_b_dot_n = dot(v_b, plane_normal);
        float f = (q_dot_n - v_a_dot_n)/(v_b_dot_n - v_a_dot_n);
        // might need to check for infs and nans here ?

        if ((0 <= f) && (f < 1)) {
            vec4 v = mix(v_a, v_b, f);
            vec4 c = mix(c_a, c_b, f);
            v = v * plane_basis;
            v[3] = 1;
            v = p3d_ModelViewProjectionMatrix * v;

            if (first) {
                first_v = v;
                first_c = c;
                first = false;
            }

            gl_Position = v;
            vertex_color = c;
            EmitVertex();
            n ++;
        }
    }
    if (3 < n) {
        gl_Position = first_v;
        vertex_color = first_c;
        EmitVertex();
    }
    EndPrimitive();
}