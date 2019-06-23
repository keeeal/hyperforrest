
import random
from itertools import combinations

from shapes3 import *

class Shape4:
    def __len__(self):
        return 4

class Plane4(Shape4):
    def __init__(self, point, normal):
        super().__init__()
        assert len(point) == len(normal) == len(self)
        self.point, self.normal = point, normal

class Tetra4(Shape4):
    def __init__(self, p1, p2, p3, p4, p5):
        super().__init__()
        self.points = p1, p2, p3, p4, p5

    def slice(self, plane):
        intersections = set()
        q, n = plane.point, plane.normal
        p_dot_n = sorted((p.dot(n), p) for p in self.points)
        q_dot_n = q.dot(n)

        for (i_dot_n, i), (j_dot_n, j) in combinations(p_dot_n, 2):
            assert i_dot_n <= j_dot_n
            if i_dot_n == q_dot_n: intersections.add(i)
            if j_dot_n == q_dot_n: intersections.add(j)
            if i_dot_n < q_dot_n < j_dot_n:
                x = i + (j-i)*(q_dot_n-i_dot_n)/(j_dot_n-i_dot_n)
                intersections.add(x)

        intersections = (Point3(p.x, p.y, p.z) for p in intersections)
        return Tetra3(*intersections)
