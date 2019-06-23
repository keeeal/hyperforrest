
import random
from itertools import combinations

from panda3d.core import Point3, Vec3, Geom, GeomNode, GeomTriangles
from panda3d.core import GeomVertexData, GeomVertexFormat, GeomVertexWriter

format = GeomVertexFormat.getV3n3cpt2()
usage = Geom.UHDynamic

class Shape3:
    def __len__(self):
        return 3

class Plane3(Shape3):
    def __init__(self, point, normal):
        super().__init__()
        assert len(point) == len(normal) == len(self)
        self.point, self.normal = point, normal

class Tetra3(Shape3):
    def __init__(self, p1, p2, p3, p4):
        super().__init__()
        self.points = p1, p2, p3, p4
        self.vertices = GeomVertexData(str(self), format, usage)
        self.triangles = GeomTriangles(usage)

        verts = GeomVertexWriter(self.vertices, 'vertex')
        norms = GeomVertexWriter(self.vertices, 'normal')
        colrs = GeomVertexWriter(self.vertices, 'color')

        for idx, (i, j, k) in enumerate(combinations(self.points, 3)):
            q = self.points[3 - idx]
            n = (j-i).cross(k-i).normalized()
            order = (i, j, k) if q.dot(n) < i.dot(n) else (i, k, j)
            r = random.random()

            for p in order:
                verts.addData3(p)
                norms.addData3(n)
                colrs.addData4(r, r, r, 1.)

        for i in range(0, self.vertices.get_num_rows(), 3):
            self.triangles.addVertices(*range(i, i + 3))

    def get_geometry(self):
        geometry = Geom(self.vertices)
        geometry.addPrimitive(self.triangles)
        return geometry

    def get_node(self):
        node = GeomNode(str(self))
        node.addGeom(self.get_geometry())
        return node
