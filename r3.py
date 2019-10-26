
from itertools import combinations

from panda3d.core import Point3, Vec3
from panda3d.core import Geom, GeomNode, GeomTriangles
from panda3d.core import GeomVertexData, GeomVertexFormat, GeomVertexWriter

from colour import white

class R3:
    def __len__(self):
        return 3

class Vertex3(R3):
    def __init__(self, point, normal=None, colour=None):
        super().__init__()
        self.point, self.normal = point, normal
        self.colour = colour if colour else white

    def __hash__(self):
        return hash(self.point)

    def __eq__(self, other):
        if not isinstance(other, Vertex3):
            return NotImplemented
        return self.point == other.point

class Geometry3(R3):
    def __init__(self, vertices, triangles):
        super().__init__()
        self.vertices, self.triangles = vertices, triangles

    def get_node(self):
        vertex_data = GeomVertexData(str(self),
            GeomVertexFormat.getV3n3c4(), Geom.UHStatic)
        triangle_data = GeomTriangles(Geom.UHStatic)

        vertex_data.setNumRows(len(self.vertices))
        # triangle_data.setNumRows(len(self.triangles))

        vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
        normal_writer = GeomVertexWriter(vertex_data, 'normal')
        colour_writer = GeomVertexWriter(vertex_data, 'color')

        P = [v.point for v in self.vertices]
        N = [v.normal for v in self.vertices]
        C = [v.colour for v in self.vertices]

        for i, j, k in self.triangles:

            n = (P[j]-P[i]).cross(P[k]-P[i]).normalized()
            for a in i, j, k:
                N[a] = n

            # OPTIMIZE: Not all normals need recalculating

            for a in i, j, k:
                vertex_writer.addData3(P[a])
                normal_writer.addData3(N[a])
                colour_writer.addData4(C[a])

        for i in range(0, vertex_data.get_num_rows(), 3):
            triangle_data.addVertices(*range(i, i+3))

        geometry = Geom(vertex_data)
        geometry.addPrimitive(triangle_data)
        node = GeomNode(str(self))
        node.addGeom(geometry)
        return node
