
from itertools import combinations

import numpy as np
from panda3d.core import Point4, Vec4

from r3 import Point3, Vertex3, Geometry3
from colour import white

class R4:
    def __len__(self):
        return 4

class Vertex4(R4):
    def __init__(self, point, normal=None, colour=None):
        super().__init__()
        self.point, self.normal = point, normal
        self.colour = colour if colour else white

    def __hash__(self):
        return hash(self.point)

    def __eq__(self, other):
        if not isinstance(other, Vertex4):
            return NotImplemented
        return self.point == other.point

class Plane4(R4):
    '''
    A hyperplane in R4.

    Args:
        origin (Point4): A point on the plane.
        normal (Vec4): A unit vector perpendicular to the plane.
        basis (list): ...
    '''

    def __init__(self, origin, normal, basis):
        super().__init__()
        self.origin, self.normal, self.basis = origin, normal, basis

    def ref(self, point):
        return Point3(
            point.dot(self.basis[0]),
            point.dot(self.basis[1]),
            point.dot(self.basis[2]))

class Geometry4(R4):
    def __init__(self, vertices, tetrahedra):
        super().__init__()
        self.vertices, self.tetrahedra = vertices, tetrahedra

    def slice(self, plane):
        P = [v.point for v in self.vertices]
        # N = [v.normal for v in self.vertices]
        C = [v.colour for v in self.vertices]

        q, n = plane.origin, plane.normal
        P_dot_n = [p.dot(n) for p in P]
        q_dot_n = q.dot(n)

        slice_points = []
        slice_triangles = []

        for tetra in self.tetrahedra:
            tetra = sorted(tetra, key=lambda i: P_dot_n[i])
            intersections = set()

            for a, b in combinations(tetra, 2):
                a_dot_n, b_dot_n = P_dot_n[a], P_dot_n[b]

                if a_dot_n == q_dot_n:
                    v = Vertex3(plane.ref(P[a]), colour=C[a])
                    intersections.add(v)

                if b_dot_n == q_dot_n:
                    v = Vertex3(plane.ref(P[b]), colour=C[b])
                    intersections.add(v)

                if a_dot_n < q_dot_n < b_dot_n:
                    f = (q_dot_n - a_dot_n)/(b_dot_n - a_dot_n)
                    p = P[a] + (P[b] - P[a])*f
                    c = C[a] + (C[b] - C[a])*f
                    v = Vertex3(plane.ref(p), colour=c)
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
