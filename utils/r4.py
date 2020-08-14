
from random import getrandbits
from itertools import combinations, product

import numpy as np
from opensimplex import OpenSimplex
from panda3d.core import GeomVertexArrayFormat, GeomVertexFormat, GeomLinesAdjacency

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
        normal :: array_like (4,) [X,Y,Z,W]
            A unit vector perpendicular to the plane.
        basis :: array_like (4,3)
            An array, the columns of which are 4-vectors defining the
            coordinate space within the hyperplane.
    '''

    def __init__(self, origin, normal, basis):
        super().__init__()
        self.origin = origin
        self.normal = normal
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
        self.data.setNumRows(len(vertices))
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


class Simplex4(Mesh4):
    '''
    A 4-simplex, the simplest 4D shape.

    Args:
        vertices :: array_like (5,4)
            The vertex postitions of the 4-simplex.
        colours :: array_like (5,4) or (1,4) (optional)
            The colours of each vertex. If a shape (1,4) is provided then all
            vertices are the same colour. Default: WHITE.
    '''

    def __init__(self, vertices, colours=None):
        assert len(vertices) == 5
        if not colours:
            colours = [WHITE]
        if len(colours) == 1:
            colours = 5*colours

        vertices = np.array(vertices)
        center = vertices.mean(axis=0)

        # groups is a list containing, for each vertex, an array of the other
        # four vertices
        groups = [np.stack(i) for i in combinations(vertices, 4)]

        # the normal for each tetrahedron is a vector perpedicular found using
        # the determinant (https://math.stackexchange.com/questions/904172) and
        # then directed away from the center
        normals = []
        for group in groups:

            normal = []
            first = group[0]
            group = group - first
            for n in range(4):
                cols = list(range(4))
                cols.remove(n)
                det = np.linalg.det(group[1:, cols])
                if n % 2:
                    det *= -1
                normal.append(det)

            normal = np.stack(normal)
            if normal.dot(first - center) < 0:
                normal *= -1
            normals += 4*[normal]

        # prepare mesh data
        vertices = np.concatenate(groups)
        normals = norm(np.stack(normals), axis=1)
        colours = [np.stack(i) for i in combinations(colours, 4)]
        colours = np.concatenate(colours)
        tetrahedra = np.arange(20).reshape(5, 4)

        super().__init__(vertices, normals, colours, tetrahedra)


class Terrain4(Mesh4):
    '''
    4D terrain procedurally generated using simplex noise.

    Args:
    '''

    def __init__(self, size, shape, scale=1, height=1, colours=None, seed=None):
        if colours is None:
            colours = [WHITE]
        if len(colours) == 1:
            colours = np.prod(shape)*colours

        if seed is None:
            seed = getrandbits(32)
        self.noise = OpenSimplex(seed).noise3d

        # TODO: Is there a better way of sampling the noise? Perhaps radially

        x = np.arange(shape[0])
        y = np.arange(shape[1])
        w = np.arange(shape[2])

        x, y, w = np.meshgrid(x, y, w)

        odd = (x + y + w) % 2 == 1
        even = (odd + 1) % 2 == 1

        odd, even = odd[:-1, :-1, :-1], even[:-1, :-1, :-1]

        z = []
        x, y, w = x.flatten(), y.flatten(), w.flatten()
        for i, j, k in zip(x, y, w):
            z.append(height*(self.noise(i/scale, j/scale, k/scale) + 1)/2)

        x = size[0] * x
        y = size[1] * y
        w = size[2] * w

        vertices = np.stack((x, y, z, w), axis=1)

        t = np.arange(np.prod(shape)).reshape(shape)

        t_0 = t[:-1, :-1, :-1][even]
        t_1 = t[:-1, :-1, 1:][even]
        t_2 = t[:-1, 1:, :-1][even]
        t_3 = t[:-1, 1:, 1:][even]
        t_4 = t[1:, :-1, :-1][even]
        t_5 = t[1:, :-1, 1:][even]
        t_6 = t[1:, 1:, :-1][even]
        t_7 = t[1:, 1:, 1:][even]

        print(t_0.shape)

        even = np.concatenate((
            np.stack((t_0, t_1, t_2, t_4), axis=1),
            np.stack((t_1, t_2, t_3, t_7), axis=1),
            np.stack((t_1, t_2, t_4, t_7), axis=1),
            np.stack((t_1, t_4, t_5, t_7), axis=1),
            np.stack((t_2, t_4, t_6, t_7), axis=1),
        ))

        t_0 = t[:-1, :-1, :-1][odd]
        t_1 = t[:-1, :-1, 1:][odd]
        t_2 = t[:-1, 1:, :-1][odd]
        t_3 = t[:-1, 1:, 1:][odd]
        t_4 = t[1:, :-1, :-1][odd]
        t_5 = t[1:, :-1, 1:][odd]
        t_6 = t[1:, 1:, :-1][odd]
        t_7 = t[1:, 1:, 1:][odd]

        odd = np.concatenate((
            np.stack((t_7, t_6, t_5, t_3), axis=1),
            np.stack((t_6, t_5, t_4, t_0), axis=1),
            np.stack((t_6, t_5, t_3, t_0), axis=1),
            np.stack((t_6, t_3, t_2, t_0), axis=1),
            np.stack((t_5, t_3, t_1, t_0), axis=1),
        ))

        tetrahedra = np.concatenate((even, odd))

        print(tetrahedra.shape)

        normals = np.prod(shape)*[(0, 0, 1, 0)]

        super().__init__(vertices, normals, colours, tetrahedra)


class Sphere4(Mesh4):
    def __init__(self, radius=1, colours=None, n=16):
        # TODO: Is there a better way of triangulating a hypersphere?

        if colours is None:
            colours = [WHITE]
        if len(colours) == 1:
            colours = (2*n**3)*colours

        theta = np.linspace(0, np.pi, n)
        phi = np.linspace(0, np.pi, n)
        omega = np.linspace(0, 2*np.pi, n)

        theta, phi, omega = np.meshgrid(theta, phi, omega)

        x = np.cos(theta)
        y = np.sin(theta)*np.cos(phi)
        z = np.sin(theta)*np.sin(phi)*np.cos(omega)
        w = np.sin(theta)*np.sin(phi)*np.sin(omega)
        x, y, z, w = x.flatten(), y.flatten(), z.flatten(), w.flatten()

        normals = np.stack((x, y, z, w), axis=1)
        vertices = radius*normals# + center

        t = np.arange(len(vertices)).reshape(theta.shape)
        t_0 = t[:-1, :-1, :-1].flatten()
        t_1 = t[:-1, :-1, 1:].flatten()
        t_2 = t[:-1, 1:, :-1].flatten()
        t_3 = t[:-1, 1:, 1:].flatten()
        t_4 = t[1:, :-1, :-1].flatten()
        t_5 = t[1:, :-1, 1:].flatten()
        t_6 = t[1:, 1:, :-1].flatten()
        t_7 = t[1:, 1:, 1:].flatten()

        tetrahedra = np.concatenate((
            np.stack((t_0, t_1, t_2, t_4), axis=1),
            np.stack((t_1, t_2, t_3, t_7), axis=1),
            np.stack((t_1, t_2, t_4, t_7), axis=1),
            np.stack((t_1, t_4, t_5, t_7), axis=1),
            np.stack((t_2, t_4, t_6, t_7), axis=1),
        ))

        super().__init__(vertices, normals, colours, tetrahedra)


class Cube4(Mesh4):
    '''
    A hypercube with arbitrary vertex locations. More like a hyper-
    quadrilateral.

    Args:
        vertices :: array_like (16,4)
            The vertex postitions of the hypercube.
        colours :: array_like (16,4) or (1,4) (optional)
            The colours of each vertex. If a shape (1,4) is provided then all
            vertices are the same colour. Default: WHITE.
    '''

    def __init__(self, vertices, colours=None):
        assert len(vertices) == 16
        if colours is None:
            colours = [WHITE]
        if len(colours) == 1:
            colours = 16*colours

        # TODO: triangulation magic here

        super().__init__(vertices, normals, colours, tetrahedra)
