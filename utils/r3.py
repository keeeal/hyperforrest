
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


class Sphere3(Mesh3):
    def __init__(self, radius, colours=[WHITE], n=100):
        pass
