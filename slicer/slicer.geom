#version 430

layout(lines_adjacency) in;
layout(triangle_strip, max_vertices=6) out;

uniform mat4 p3d_ModelViewProjectionMatrix;

uniform vec4 plane_origin;
uniform vec4 plane_normal;
uniform mat4 plane_basis;

in VertexData
{
    vec4 vertex;
    vec4 normal;
    vec4 colour;
} vertex_data[];

out vec4 vertex_normal;
out vec4 vertex_colour;

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

    int intersections = 0;
    for (int i = 0; i < 6; i ++)
    {
        vec4 v_a = vertex_data[a[i]].vertex;
        vec4 v_b = vertex_data[b[i]].vertex;
        vec4 n_a = vertex_data[a[i]].normal;
        vec4 n_b = vertex_data[b[i]].normal;
        vec4 c_a = vertex_data[a[i]].colour;
        vec4 c_b = vertex_data[b[i]].colour;

        // calculate f, the fraction of the line connecting a and b that lies
        // to a's side of the plane
        float v_a_dot_n = dot(v_a, plane_normal);
        float v_b_dot_n = dot(v_b, plane_normal);
        float f = (q_dot_n - v_a_dot_n)/(v_b_dot_n - v_a_dot_n);

        if ((0 <= f) && (f < 1)) {

            // find new vertex position
            vec4 v = mix(v_a, v_b, f);
            v = v * plane_basis;
            v.w = 1;
            v = p3d_ModelViewProjectionMatrix * v;

            // find new normal
            vec4 n = mix(n_a, n_b, f);
            n = n * plane_basis;
            n.w = 0;

            // find new colour
            vec4 c = mix(c_a, c_b, f);

            if (intersections == 0) {
                first_v = v;
                first_n = n;
                first_c = c;
            }

            intersections ++;
            gl_Position = v;
            vertex_colour = c;
            vertex_normal = n;
            EmitVertex();
        }
    }
    if (3 < intersections) {
        gl_Position = first_v;
        vertex_colour = first_c;
        vertex_normal = first_n;
        EmitVertex();
    }
    EndPrimitive();
}