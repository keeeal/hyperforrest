
from itertools import combinations
# from functools import total_ordering

from panda3d.core import Point4, Vec4

from r3 import Vertex3, Geometry3
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
    def __init__(self, point, normal):
        super().__init__()
        assert len(point) == len(normal) == len(self)
        self.point, self.normal = point, normal

class Geometry4(R4):
    def __init__(self, vertices, tetrahedra):
        super().__init__()
        self.vertices, self.tetrahedra = vertices, tetrahedra

    def slice(self, plane):
        P = [v.point for v in self.vertices]
        N = [v.normal for v in self.vertices]
        C = [v.colour for v in self.vertices]

        q, n = plane.point, plane.normal
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
                    x = Vertex3(P[a].get_xyz(), colour=C[a])
                    intersections.add(x)
                if b_dot_n == q_dot_n:
                    x = Vertex3(P[b].get_xyz(), colour=C[b])
                    intersections.add(x)
                if a_dot_n < q_dot_n < b_dot_n:
                    f = (q_dot_n - a_dot_n)/(b_dot_n - a_dot_n)
                    p = P[a] + (P[b] - P[a])*f
                    c = C[a] + (C[b] - C[a])*f
                    x = Vertex3(p.get_xyz(), colour=c)
                    intersections.add(x)

            if len(intersections) > 4:
                print(len(intersections))
                raise ValueError

            elif len(intersections) > 2:
                m = len(slice_points) - 1
                slice_points.extend(intersections)
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

# class Rectangle4(Geometry4):
#     def __init__(self, center, x_size, y_size, z_size, w_size):
#         assert len(points) == 16
#         tetrahedra = [[i for i in range(5) if i is not j] for j in range(5)]
#         super().__init__(points, tetrahedra)
