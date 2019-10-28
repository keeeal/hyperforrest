
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
        self.normal = np.array(normal)
        self.basis = np.array(basis)

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
        self.radii = self.vertices[self.tetrahedra] - np.stack(4*(m,), axis=1)
        self.radii = np.max(np.linalg.norm(self.radii, axis=2), axis=1)

        self.mesh3 = 1

    def slice(self, plane):
        slice_vertices, slice_normals = [], []
        slice_colours, slice_triangles = [], []

        V_dot_n = self.vertices.dot(plane.normal)
        M_dot_n = self.means.dot(plane.normal)
        q_dot_n = plane.origin.dot(plane.normal)

        for t, m_dot_n, r in zip(self.tetrahedra, M_dot_n, self.radii):
            if r < abs(m_dot_n - q_dot_n):
                continue

            intersections = set()
            t = sorted(t, key=lambda i: V_dot_n[i])

            for a, b in combinations(t, 2):
                a_dot_n, b_dot_n = V_dot_n[a], V_dot_n[b]

                if a_dot_n == q_dot_n:
                    v = plane.transform(self.vertices[a])
                    c = self.colours[a]
                    intersections.add(v)

                if b_dot_n == q_dot_n:
                    v = plane.transform(self.vertices[b])
                    c = self.colours[b]
                    intersections.add(v)

                if a_dot_n < q_dot_n < b_dot_n:
                    f = (q_dot_n - a_dot_n)/(b_dot_n - a_dot_n)
                    v = self.vertices[a] + (self.vertices[b] - self.vertices[a])*f
                    c = self.colours[a] + (self.colours[b] - self.colours[a])*f
                    v = plane.transform(v)
                    intersections.add(v)

            if len(intersections) > 4:
                raise ValueError

            elif len(intersections) > 2:
                slice_points.extend(intersections)
                m = len(slice_points) - 1

                for i, j, k in combinations(range(len(intersections)), 3):
                    slice_triangles.append((m-i, m-j, m-k))
                    slice_triangles.append((m-i, m-k, m-j))

        if slice_triangles:
            return Geometry3(slice_points, slice_triangles)

class Simplex4(Geometry4):
    def __init__(self, vertices):
        assert len(vertices) == 5
        tetrahedra = list(combinations(range(5), 4))
        super().__init__(vertices, tetrahedra)

class Floor4(Geometry4):
    def __init__(self, size, height=0, sigma=1, colour=white):
        vertices, tetrahedra = [], []

        for x in range(0, size[0]):
            for y in range(0, size[1]):
                for w in range(0, size[2]):
                    z = np.random.normal(height, sigma)
                    vertices.append(Vertex4(Point4(x, y, z, w), colour=colour))

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

        super().__init__(vertices, tetrahedra)

class Sphere4(Geometry4):
    def __init__(self, center, radius, colour=white, n=8):
        vertices, tetrahedra = [], []

        for theta in np.linspace(0, np.pi, n):
            for phi in np.linspace(0, np.pi, n):
                for omega in np.linspace(0, 2*np.pi, 2*n):
                    x = radius*np.cos(theta)
                    y = radius*np.sin(theta)*np.cos(phi)
                    z = radius*np.sin(theta)*np.sin(phi)*np.cos(omega)
                    w = radius*np.sin(theta)*np.sin(phi)*np.sin(omega)
                    vertices.append(Vertex4(Point4(x, y, z, w), colour=colour))

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

        super().__init__(vertices, tetrahedra)
