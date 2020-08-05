
import torch
import numpy as np

from panda3d.core import Geom, GeomNode, GeomTriangles
from panda3d.core import GeomVertexData, GeomVertexFormat, GeomVertexWriter

from utils.colour import WHITE
from utils.math import *


class Mesh3:
    '''
    Base class for meshes in R3.

    Args:
        vertices :: array_like (N,3)
            A list of points in the mesh.
        normals :: array_like (N,3)
            A list of vectors perpendicular to the mesh at each vertex.
        colours :: array_like (N,4)
            A list of colours at each vertex.
        triangles :: array_like (M,3)
            A list of indices that group vertices into triangles.
    '''

    def __init__(self, vertices, normals, colours, triangles):
        super().__init__()

        self.vertices = tensor(vertices, dtype=torch.float)
        self.normals = tensor(normals, dtype=torch.float)
        self.colours = tensor(colours, dtype=torch.float)
        self.triangles = tensor(triangles, dtype=torch.long)
        self.device = None

    def to(self, device):
        self.device = device
        self.vertices = self.vertices.to(device)
        self.normals = self.normals.to(device)
        self.colours = self.colours.to(device)
        self.triangles = self.triangles.to(device)
        return self

    def get_node(self):
        vertices = self.vertices.cpu().numpy()
        normals = self.normals.cpu().numpy()
        colours = self.colours.cpu().numpy()
        triangles = self.triangles.cpu().numpy()

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


class Sphere3(Mesh3):
    def __init__(self, radius, center=None, colours=None, n=8):

        if center is None:
            center = [0, 0, 0]
        if isinstance(center, torch.Tensor):
            center = center.cpu().numpy()

        if colours is None:
            colours = [WHITE]
        if len(colours) == 1:
            colours = (2*n**2)*colours
        colours = tensor(colours)

        theta = torch.linspace(0, np.pi, n)#.to(self.device)
        phi = torch.linspace(0, 2*np.pi, n)#.to(self.device)
        theta, phi = torch.meshgrid(theta, phi)

        x = theta.cos()
        y = theta.sin()*phi.cos()
        z = theta.sin()*phi.sin()
        x, y, z = x.flatten(), y.flatten(), z.flatten()

        normals = torch.stack((x, y, z), axis=1)
        vertices = radius*normals + center

        t = torch.arange(len(vertices)).view(theta.shape)
        t_1 = t[:-1, :-1].flatten()
        t_2 = t[1:, :-1].flatten()
        t_3 = t[:-1, 1:].flatten()
        t_4 = t[1:, 1:].flatten()

        triangles = torch.cat((
            torch.stack((t_1, t_2, t_3), axis=1),
            torch.stack((t_4, t_3, t_2), axis=1)
        ))

        super().__init__(vertices, normals, colours, triangles)
