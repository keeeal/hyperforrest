
from random import getrandbits, choices, sample
from itertools import combinations, product

import numpy as np
from numpy.random import random as rand
from scipy.spatial import Delaunay, ConvexHull
from opensimplex import OpenSimplex
from panda3d.core import *

from utils.colour import WHITE
from utils.math import *
from utils.r3 import *

array4 = GeomVertexArrayFormat()
array4.addColumn("vertex", 4, Geom.NTFloat32, Geom.COther)
array4.addColumn("normal", 4, Geom.NTFloat32, Geom.COther)
array4.addColumn("colour", 4, Geom.NTFloat32, Geom.CColor)

format4 = GeomVertexFormat()
format4.addArray(array4)
format4 = GeomVertexFormat.registerFormat(format4)


class Plane4:
    '''
    A hyperplane in R4.

    Args:
        origin :: array_like (4,) [X,Y,Z,W]
            A point on the plane.
        basis :: array_like (4,4)
            An array defining the coordinate space within the plane.
    '''

    def __init__(self, origin, basis):
        super().__init__()
        self.origin = origin
        self.basis = basis


class Mesh4:
    '''
    Base class for meshes in R4.

    Args:
        vertices :: array_like (N,4)
            A list of vertex positions in the mesh.
        normals :: array_like (N,4)
            Vectors perpendicular to the mesh at each vertex.
        colours :: array_like (N,4)
            The colour of each vertex.
        tetrahedra :: array_like (M,4)
            A list of indices that group vertices into 4-simplices.
    '''

    def __init__(self, vertices, normals, colours, tetrahedra):
        super().__init__()
        self.n_vertices = len(vertices)
        self.n_tetrahedra = len(tetrahedra)

        self.data = GeomVertexData(repr(self), format4, Geom.UHDynamic)
        self.data.setNumRows(self.n_vertices)
        self.prim = GeomLinesAdjacency(Geom.UHDynamic)

        vertex_writer = GeomVertexWriter(self.data, 'vertex')
        normal_writer = GeomVertexWriter(self.data, 'normal')
        colour_writer = GeomVertexWriter(self.data, 'colour')

        for vertex in vertices:
            vertex_writer.addData4(*vertex)

        for normal in normals:
            normal_writer.addData4(*normal)

        for colour in colours:
            colour_writer.addData4(*colour)

        for tetra in tetrahedra:
            self.prim.addVertices(*tetra)
            self.prim.closePrimitive()

        self.geom = Geom(self.data)
        self.geom.addPrimitive(self.prim)
        self.node = GeomNode(repr(self))
        self.node.addGeom(self.geom)

    def __len__(self):
        return len(self.n_vertices)


class Hull4(Mesh4):
    '''
    A convex hull in R4.

    Args:
        vertices :: array_like (N,4)
            A list of vertex positions in the mesh.
        colours :: array_like (N,4)
            The colour of each vertex.
    '''

    def __init__(self, vertices, colours=[WHITE]):
        if len(colours) == 1:
            colours = len(vertices) * colours

        vertices = np.array(vertices)
        colours = np.array(colours)
        center = vertices.mean(axis=0)

        # calculate tetrahedra and a normal for each
        tetrahedra = ConvexHull(vertices).simplices
        normals = get_tetra_norms(vertices, tetrahedra, center)

        # convert to long format (repeated vertices)
        vertices = np.concatenate([vertices[t] for t in tetrahedra])
        normals = np.repeat(normals, 4, axis=0)
        colours = np.concatenate([colours[t] for t in tetrahedra])
        tetrahedra = np.arange(len(vertices)).reshape(-1, 4)


        super().__init__(vertices, normals, colours, tetrahedra)


class Cube4(Hull4):
    '''
    A hypercube in R4.

    Args:
        size :: array_like (4,) [X,Y,Z,W]
            The dimensions of the hypercube.
        colours :: array_like (16,4) or (1,4) (optional)
            The colours of each vertex. If a shape (1,4) is provided then all
            vertices are the same colour. Default: WHITE.
    '''

    def __init__(self, size, colours=[WHITE]):
        if len(colours) == 1:
            colours = 16*colours

        vertices = list(product(*[(-i/2, i/2) for i in size]))
        super().__init__(vertices, colours)


class Sphere4(Hull4):
    '''
    A hypersphere in R4 triangulated randomly.

    Args:
        radius :: float
            The radius of the hypersphere.
        colours :: array_like (N,4) or (1,4) (optional)
            The colours of each vertex. If a shape (1,4) is provided then all
            vertices are the same colour. Default: WHITE.
        n :: int
            The number of vertices used to triangulate the sphere. More
            vertices results in a smoother sphere but may run slower.
    '''

    def __init__(self, radius, colours=[WHITE], n=1000):
        if len(colours) == 1:
            colours = n*colours

        # TODO: There are better ways of sampling this. See
        # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

        vertices = []
        i = int(1000*rand())
        while len(vertices) < n:
            vertex = 2 * np.array(halton([i], (2, 3, 5, 7))[0]) - 1
            if length(vertex) < 1:
                vertices.append(radius * norm(vertex))
            i += 1

        vertices = np.stack(vertices)
        super().__init__(vertices, colours)


class Terrain4(Mesh4):
    '''
    4D terrain procedurally generated using simplex noise.

    Args:
    '''

    def __init__(self, size, frequency=1, height=1, colours=[WHITE], seed=None, n=1000):
        if len(colours) == 1:
            colours = n * colours

        if seed is None:
            seed = getrandbits(32)

        self.noise = OpenSimplex(seed).noise3d
        vertices = halton(range(n), (2, 3, 5))

        # set corners
        c = 8
        for i, corner in enumerate(product(*3*[(0, 1)])):
            if i < len(vertices):
                vertices[i] = corner

        # set edges
        e = int(12*n**(1/3))
        for i in range(c, c + e):
            if i < len(vertices):
                dims = sample(range(3), k=1)
                vals = choices(range(2), k=1)
                for dim, val in zip(dims, vals):
                    vertices[i][dim] = val

        # set faces
        f = int(6*n**(2/3))
        for i in range(c + e, c + e + f):
            if i < len(vertices):
                dims = sample(range(3), k=2)
                vals = choices(range(2), k=2)
                for dim, val in zip(dims, vals):
                    vertices[i][dim] = val

        vertices = size * np.array(vertices)
        colours = np.array(colours)

        tetrahedra = Delaunay(vertices).simplices
        z = [height*self.noise(x, y, w) for x, y, w in frequency*vertices]
        vertices = np.insert(vertices, 2, z, axis=1)

        center = np.array([0, 0, -np.inf, 0])
        normals = get_tetra_norms(vertices, tetrahedra, center)

        # convert to long format (repeated vertices)
        vertices = np.concatenate([vertices[t] for t in tetrahedra])
        normals = np.repeat(normals, 4, axis=0)
        colours = np.concatenate([colours[t] for t in tetrahedra])
        tetrahedra = np.arange(len(vertices)).reshape(-1, 4)

        super().__init__(vertices, normals, colours, tetrahedra)
