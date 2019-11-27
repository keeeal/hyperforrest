
from itertools import combinations

import numpy as np
from panda3d.core import Point4, Vec4

from r3 import Point3, Vertex3, Geometry3
from colour import white

class R4:
    '''
    Base class in R4.
    '''

class Plane4(R4):
    '''
    A hyperplane in R4.

    Args:
        origin :: array_like (4,) [X,Y,Z,W]
            A point on the plane.
        normal :: array_like (4,) [X,Y,Z,W]
            A unit vector perpendicular to the plane.
        basis :: array_like (4,3)
            An array, the columns of which are 4-vectors defining the
            coordinate space within the hyperplane.
    '''

    def __init__(self, origin, normal, basis):
        super().__init__()
        self.origin = np.array(origin)
        self.normal = np.array(normal)/np.linalg.norm(normal)
        self.basis = np.array(basis)/np.linalg.norm(basis, axis=1)[:,np.newaxis]

    def transform(self, points):
        '''
        Transform points (R4 -> R3) into the coordinate space of the plane.

        Args:
            points :: array_like (N,4)
                A list of points to be transformed.
        '''
        return np.array(points).dot(self.basis)

class Mesh4(R4):
    '''
    Base class for meshes in R4.

    Args:
        vertices :: array_like (N,4)
            A list of points in the mesh.
        normals :: array_like (N,4)
            A list of vectors perpendicular to the mesh at each vertex.
        colours :: array_like (N,4)
            A list of colours at each vertex.
        tetrahedra :; array_like (M,4)
            A list of indices that group vertices into 4-simplices.
    '''

    def __init__(self, vertices, normals, colours, tetrahedra):
        super().__init__()
        self.vertices = np.array(vertices)
        self.normals = np.array(normals)
        self.colours = np.array(colours)
        self.tetrahedra = np.array(tetrahedra)

        self.means = np.mean(self.vertices[self.tetrahedra], axis=1)
        self.radii = self.vertices[self.tetrahedra] - np.stack(4*(self.means,), axis=1) ## OPTIMIZE: use np.newaxis
        self.radii = np.max(np.linalg.norm(self.radii, axis=2), axis=1)

        self._a_idx = np.arange(4) - 1

    def slice(self, plane):
        vertices, normals, colours, triangles = [], [], [], []

        V_dot_n = self.vertices.dot(plane.normal)
        M_dot_n = self.means.dot(plane.normal)
        q_dot_n = plane.origin.dot(plane.normal)

        nearby = np.abs(M_dot_n - q_dot_n) < self.radii

        b = self.tetrahedra[nearby]
        a = b[:, self._a_idx]

        v_a_dot_n, v_b_dot_n = V_dot_n[a], V_dot_n[b]
        f = (q_dot_n - v_a_dot_n)/(v_b_dot_n - v_a_dot_n)
        f[f == np.nan] = 0

        intersection = (0 <= f) * (f < 1)
        intersection_count = np.sum(intersection, axis=1)

        print(f)

        tris = intersection_count == 3
        a_tris, b_tris = a[tris], b[tris]
        tris_mask = intersection[tris]
        f_tris = f[tris][tris_mask][:,np.newaxis]

        v_a_tris = self.vertices[a_tris][tris_mask]
        v_b_tris = self.vertices[b_tris][tris_mask]
        v_tris = v_a_tris + f_tris*(v_b_tris - v_a_tris)

        n_a_tris = self.normals[a_tris][tris_mask]
        n_b_tris = self.normals[b_tris][tris_mask]
        n_tris = n_a_tris + f_tris*(n_b_tris - n_a_tris)

        c_a_tris = self.colours[a_tris][tris_mask]
        c_b_tris = self.colours[b_tris][tris_mask]
        c_tris = c_a_tris + f_tris*(c_b_tris - c_a_tris)

        print(tris)

        # quads = intersection_count == 4
        # a_quads, b_quads = a[quads], b[quads]
        # quads_mask = intersection[quads]
        # f_quads = f[quads][quads_mask][:,np.newaxis]
        #
        # v_a_quads = self.vertices[a_quads][quads_mask]
        # v_b_quads = self.vertices[b_quads][quads_mask]
        # v_quads = v_a_quads + f_quads*(v_b_quads - v_a_quads)
        #
        # n_a_quads = self.normals[a_quads][quads_mask]
        # n_b_quads = self.normals[b_quads][quads_mask]
        # n_quads = n_a_quads + f_quads*(n_b_quads - n_a_quads)
        #
        # c_a_quads = self.colours[a_quads][quads_mask]
        # c_b_quads = self.colours[b_quads][quads_mask]
        # c_quads = c_a_quads + f_quads*(c_b_quads - c_a_quads)

        # print(self.vertices[b[tris]][i[tris]])
        # print(i[tris])

        # print(sum(nearby)/len(nearby))
        # print(np.min(self.radii), np.mean(self.radii), np.max(self.radii))

        # for t in t_nearby:
        #
        #     v_a, v_b = self.vertices[a], self.vertices[b]
        #     v_a_dot_n, v_b_dot_n = V_dot_n[a], V_dot_n[b]
        #     f = (q_dot_n - v_a_dot_n)/(v_b_dot_n - v_a_dot_n)
        #     f[f == np.nan] = 0
        #
        #     for i in range(4):
        #         a_dot_n, b_dot_n = V_dot_n[i-1], V_dot_n[i]
        #
        #         f = 0 if a_dot_n == b_dot_n \
        #             else (q_dot_n - a_dot_n)/(b_dot_n - a_dot_n)
        #
        #         if a_dot_n == q_dot_n:
        #             v = plane.transform(self.vertices[a])
        #             c = self.colours[a]
        #             intersections.add(v)
        #
        #         if b_dot_n == q_dot_n:
        #             v = plane.transform(self.vertices[b])
        #             c = self.colours[b]
        #             intersections.add(v)
        #
        #         if a_dot_n < q_dot_n < b_dot_n:
        #             f = (q_dot_n - a_dot_n)/(b_dot_n - a_dot_n)
        #             v = self.vertices[a] + (self.vertices[b] - self.vertices[a])*f
        #             c = self.colours[a] + (self.colours[b] - self.colours[a])*f
        #             v = plane.transform(v)
        #             intersections.add(v)
        #
        #     if len(intersections) > 4:
        #         raise ValueError
        #
        #     elif len(intersections) > 2:
        #         slice_points.extend(intersections)
        #         m = len(slice_points) - 1
        #
        #         for i, j, k in combinations(range(len(intersections)), 3):
        #             slice_triangles.append((m-i, m-j, m-k))
        #             slice_triangles.append((m-i, m-k, m-j))

        return Geometry3(vertices, normals, colours, triangles)

class Simplex4(Mesh4):
    def __init__(self, vertices, colours):
        assert len(vertices) == 5 and len(colours) == 5
        vertices = np.array(vertices)
        center = np.mean(vertices, axis=0)
        normals = vertices - center
        normals = normals/np.linalg.norm(normals, axis=1)[:,np.newaxis]
        tetrahedra = np.array(list(combinations(range(5), 4)))
        super().__init__(vertices, normals, colours, tetrahedra)

class Floor4(Mesh4):
    def __init__(self, size, height=0, sigma=1, colour=white):
        vertices, normals, colours, tetrahedra = [], [], [], []

        for x in range(0, size[0]):
            for y in range(0, size[1]):
                for w in range(0, size[2]):
                    z = np.random.normal(height, sigma)
                    vertices.append((x, y, z, w))

        for i in range(0, size[0] - 1):
            for j in range(0, size[1] - 1):
                for k in range(0, size[2] - 1):
                    if (i + j + k) % 2:
                        tetrahedra.append(list(np.ravel_multi_index(((i,i+1,i,i),(j,j,j+1,j),(k,k,k,k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((i+1,i+1,i,i),(j+1,j,j+1,j),(k+1,k,k,k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((i+1,i,i+1,i+1),(j,j+1,j+1,j+1),(k,k,k,k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((i,i+1,i+1,i+1),(j,j,j,j+1),(k+1,k,k+1,k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((i,i,i,i+1),(j+1,j,j+1,j+1),(k,k+1,k+1,k+1)), size)))
                    else:
                        tetrahedra.append(list(np.ravel_multi_index(((i,i+1,i+1,i+1),(j,j,j+1,j),(k,k,k,k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((i,i,i+1,i),(j,j+1,j+1,j+1),(k,k,k,k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((i,i,i+1,i),(j,j,j,j+1),(k,k+1,k+1,k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((i,i+1,i+1,i),(j,j+1,j,j+1),(k,k,k+1,k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((i+1,i+1,i+1,i),(j+1,j+1,j,j+1),(k+1,k,k+1,k+1)), size)))

        super().__init__(vertices, normals, colours, tetrahedra)

class Sphere4(Mesh4):
    def __init__(self, center, radius, colour=white, n=8):
        center = np.array(center)
        vertices, normals, colours, tetrahedra = [], [], [], []

        for theta in np.linspace(0, np.pi, n):
            for phi in np.linspace(0, np.pi, n):
                for omega in np.linspace(0, 2*np.pi, 2*n):
                    x = np.cos(theta)
                    y = np.sin(theta)*np.cos(phi)
                    z = np.sin(theta)*np.sin(phi)*np.cos(omega)
                    w = np.sin(theta)*np.sin(phi)*np.sin(omega)
                    vertices.append(center + radius*np.array((x, y, z, w)))
                    normals.append(np.array((x, y, z, w)))
                    colours.append(colour)

        for i in range(n-1):
            for j in range(n-1):
                for k in range(2*n-1):
                    if (i + j + k) % 2:
                        tetrahedra.append(list(np.ravel_multi_index(((i,i+1,i,i),(j,j,j+1,j),(k,k,k,k+1)), (n,n,2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(((i+1,i+1,i,i),(j+1,j,j+1,j),(k+1,k,k,k+1)), (n,n,2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(((i+1,i,i+1,i+1),(j,j+1,j+1,j+1),(k,k,k,k+1)), (n,n,2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(((i,i+1,i+1,i+1),(j,j,j,j+1),(k+1,k,k+1,k+1)), (n,n,2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(((i,i,i,i+1),(j+1,j,j+1,j+1),(k,k+1,k+1,k+1)), (n,n,2*n))))
                    else:
                        tetrahedra.append(list(np.ravel_multi_index(((i,i+1,i+1,i+1),(j,j,j+1,j),(k,k,k,k+1)), (n,n,2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(((i,i,i+1,i),(j,j+1,j+1,j+1),(k,k,k,k+1)), (n,n,2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(((i,i,i+1,i),(j,j,j,j+1),(k,k+1,k+1,k+1)), (n,n,2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(((i,i+1,i+1,i),(j,j+1,j,j+1),(k,k,k+1,k+1)), (n,n,2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(((i+1,i+1,i+1,i),(j+1,j+1,j,j+1),(k+1,k,k+1,k+1)), (n,n,2*n))))

        super().__init__(vertices, normals, colours, tetrahedra)
