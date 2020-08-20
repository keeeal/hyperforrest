#version 430

layout(lines_adjacency) in;
layout(triangle_strip, max_vertices=6) out;

uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;

uniform vec4 plane_origin;
uniform mat4 plane_basis;

in VertexData
{
    vec4 vertex;
    vec4 normal;
    vec4 colour;
} vertex_data[];

out FragmentData
{
    vec4 vertex;
    vec3 normal;
    vec4 colour;
} fragment_data;

void main()
{
    vec4 v_dot_n;
    for (int i = 0; i < 4; i ++)
    {
        v_dot_n[i] = dot(vertex_data[i].vertex, plane_basis[3]);
    }
    float q_dot_n = dot(plane_origin, plane_basis[3]);

    // view plane culling
    vec4 _q = {q_dot_n, q_dot_n, q_dot_n, q_dot_n};
    if (all(lessThan(v_dot_n, _q))) return;
    if (all(greaterThan(v_dot_n, _q))) return;

    // a and b index each pair of points in the tetrahedron
    int a[6] = {0, 0, 1, 1, 2, 3};
    int b[6] = {1, 2, 2, 3, 3, 0};

    vec4 first_vertex;
    vec3 first_normal;
    vec4 first_colour;

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
        float v_a_dot_n = dot(v_a, plane_basis[3]);
        float v_b_dot_n = dot(v_b, plane_basis[3]);
        float f = (q_dot_n - v_a_dot_n)/(v_b_dot_n - v_a_dot_n);

        if ((0 <= f) && (f < 1)) {

            // find new vertex position
            vec4 v = mix(v_a, v_b, f);
            v = v * plane_basis;

            // find new normal
            vec4 n = mix(n_a, n_b, f);
            n = n * plane_basis;

            // find new colour
            vec4 c = mix(c_a, c_b, f);

            fragment_data.vertex = p3d_ModelViewMatrix * vec4(v.xyz, 1);
            fragment_data.normal = p3d_NormalMatrix * n.xyz;
            fragment_data.colour = c;

            if (intersections == 0) {
                first_vertex = fragment_data.vertex;
                first_normal = fragment_data.normal;
                first_colour = fragment_data.colour;
            }

            gl_Position = p3d_ProjectionMatrix * fragment_data.vertex;
            intersections ++;

            EmitVertex();
        }
    }
    if (3 < intersections) {

        fragment_data.vertex = first_vertex;
        fragment_data.normal = first_normal;
        fragment_data.colour = first_colour;

        gl_Position = p3d_ProjectionMatrix * fragment_data.vertex;

        EmitVertex();
    }
    EndPrimitive();
}