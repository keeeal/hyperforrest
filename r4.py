
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
    def __init__(self, origin, normal, base_x, base_y, base_z):
        super().__init__()
        self.origin, self.normal = origin, normal
        self.base_x, self.base_y, self.base_z = base_x, base_y, base_z

    def ref(self, point):
        return Point3(
            point.dot(self.base_x),
            point.dot(self.base_y),
            point.dot(self.base_z))

class Geometry4(R4):
    def __init__(self, vertices, tetrahedra):
        super().__init__()
        self.vertices, self.tetrahedra = vertices, tetrahedra

    def slice(self, plane):
        P = [v.point for v in self.vertices]
        N = [v.normal for v in self.vertices]
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

        for x in range(0, size[0] - 1):
            for y in range(0, size[1] - 1):
                for w in range(0, size[2] - 1):
                    if (x + y + w) % 2:
                        tetrahedra.append(list(np.ravel_multi_index(((x,x+1,x,x),(y,y,y+1,y),(w,w,w,w+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((x+1,x+1,x,x),(y+1,y,y+1,y),(w+1,w,w,w+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((x+1,x,x+1,x+1),(y,y+1,y+1,y+1),(w,w,w,w+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((x,x+1,x+1,x+1),(y,y,y,y+1),(w+1,w,w+1,w+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((x,x,x,x+1),(y+1,y,y+1,y+1),(w,w+1,w+1,w+1)), size)))
                    else:
                        tetrahedra.append(list(np.ravel_multi_index(((x,x+1,x+1,x+1),(y,y,y+1,y),(w,w,w,w+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((x,x,x+1,x),(y,y+1,y+1,y+1),(w,w,w,w+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((x,x,x+1,x),(y,y,y,y+1),(w,w+1,w+1,w+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((x,x+1,x+1,x),(y,y+1,y,y+1),(w,w,w+1,w+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(((x+1,x+1,x+1,x),(y+1,y+1,y,y+1),(w+1,w,w+1,w+1)), size)))

        super().__init__(vertices, tetrahedra)
