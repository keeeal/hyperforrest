
from panda3d.core import Geom, GeomNode, GeomTriangles
from panda3d.core import GeomVertexData, GeomVertexFormat, GeomVertexWriter

from utils.colour import WHITE


class R3:
    '''
    Base class in R3.
    '''

    def __len__(self):
        return 3


class Geometry3(R3):
    def __init__(self, vertices, normals, colours, triangles):
        super().__init__()
        self.vertices = vertices.numpy()
        self.normals = normals.numpy()
        self.colours = colours.numpy()
        self.triangles = triangles.numpy()

        # print(vertices)
        # print(normals)
        # print(colours)

    def get_node(self):
        vertex_data = GeomVertexData(repr(self),
            GeomVertexFormat.getV3n3c4(), Geom.UHStatic)
        vertex_data.setNumRows(len(self.vertices))
        triangle_data = GeomTriangles(Geom.UHStatic)

        vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
        normal_writer = GeomVertexWriter(vertex_data, 'normal')
        colour_writer = GeomVertexWriter(vertex_data, 'color')

        for vertex in self.vertices:
            vertex_writer.addData3(*vertex)

        for normal in self.normals:
            normal_writer.addData3(*normal)

        for colour in self.colours:
            colour_writer.addData4(*colour)

        for triangle in self.triangles:
            triangle_data.addVertices(*triangle)

        geometry = Geom(vertex_data)
        geometry.addPrimitive(triangle_data)
        node = GeomNode(repr(self))
        node.addGeom(geometry)
        return node
