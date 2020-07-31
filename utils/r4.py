
from itertools import combinations

import torch
import numpy as np

from utils.r3 import Geometry3
from utils.colour import WHITE
from utils.math import *


class R4:
    '''
    Base class in R4.
    '''

    def __len__(self):
        return 4


class Plane4(R4):
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
        self.origin = tensor(origin, dtype=torch.float)
        self.normal = norm(tensor(normal, dtype=torch.float))
        self.basis = norm(tensor(basis, dtype=torch.float))

    def to(device):
        self.origin = self.origin.to(device)
        self.normal = self.normal.to(device)
        self.basis = self.basis.to(device)

    def transform(self, points):
        '''
        Transform points (R4 -> R3) into the coordinate space of the plane.

        Args:
            points :: array_like (N,4)
                A list of points to be transformed.
        '''
        return points.matmul(self.basis)


class Mesh4(R4):
    '''
    Base class for meshes in R4.

    Args:
        vertices :: array_like (N,4)
            A list of points in the mesh.
        normals :: array_like (N,4)
            A list of vectors perpendicular to the mesh at each vertex.
        colours :: array_like (N,4)
            A list of colours at each vertex.
        tetrahedra :; array_like (M,4)
            A list of indices that group vertices into 4-simplices.
    '''

    def __init__(self, vertices, normals, colours, tetrahedra):
        super().__init__()
        self.vertices = tensor(vertices, dtype=torch.float)
        self.normals = tensor(normals, dtype=torch.float)
        self.colours = tensor(colours, dtype=torch.float)
        self.tetrahedra = tensor(tetrahedra, dtype=torch.long)

        # The average location and radial extent of each tetrahedron is stored
        # to make slicing more efficient.
        self.means = self.vertices[self.tetrahedra].mean(axis=1, keepdims=True)
        self.radii = self.vertices[self.tetrahedra] - self.means
        self.radii = length(self.radii, axis=2).max(axis=1)[0]
        self.means = self.means.squeeze(1)

    def slice(self, plane):

        # calculate how far from the plane each point is
        V_dot_n = self.vertices.matmul(plane.normal)
        M_dot_n = self.means.matmul(plane.normal)
        q_dot_n = plane.origin.matmul(plane.normal)

        # only consider tetrahedra near the plane
        nearby = (M_dot_n - q_dot_n).abs() < self.radii

        # a and b are each pair of points in each tetrahedron
        # BUG: THIS IS NOT EVERY COMBINATION OF
        # POINTS, ONLY 4 OF THE 6 POSSIBLE!
        a = self.tetrahedra[nearby]
        b = a[:, (1, 2, 3, 0)]

        # calculate f: the fraction of the line connecting a and b that lies
        # to one side of the plane
        v_a_dot_n, v_b_dot_n = V_dot_n[a], V_dot_n[b]
        f = (q_dot_n - v_a_dot_n)/(v_b_dot_n - v_a_dot_n)
        f[v_a_dot_n == q_dot_n] = 0
        # f[v_b_dot_n == q_dot_n] = 1

        # from f, determine whether an intersection has occured and, if so,
        # how many times each tetrahedron has been intersected
        intersection = (0 <= f) * (f < 1)
        intersection_count = intersection.sum(axis=1)

        print()
        print(a)
        print(b)
        print(f)
        print(intersection_count)

        # prepare 3D data structures
        vertices = torch.empty((0, 3), dtype=torch.float)
        normals = torch.empty((0, 3), dtype=torch.float)
        colours = torch.empty((0, 4), dtype=torch.float)
        triangles = torch.empty((0, 3), dtype=torch.long)

        # if the edges of a tetrahedron are intersected either 3 or 4 times,
        # it is visible
        for n in 3, 4:

            mask = intersection_count == n
            _a, _b, _f = a[mask], b[mask], f[mask]
            _i = intersection[mask]
            _f_i = _f[_i].unsqueeze(1)

            # calculate 4-vertices of the triangles
            _v_a_i = self.vertices[_a][_i]
            _v_b_i = self.vertices[_b][_i]
            _v_i = _v_a_i + _f_i*(_v_b_i - _v_a_i)

            # calculate 4-normals of the triangles
            _n_a_i = self.normals[_a][_i]
            _n_b_i = self.normals[_b][_i]
            _n_i = _n_a_i + _f_i*(_n_b_i - _n_a_i)

            # calculate colours of the triangles
            _c_a_i = self.colours[_a][_i]
            _c_b_i = self.colours[_b][_i]
            _c_i = _c_a_i + _f_i*(_c_b_i - _c_a_i)

            new_tris = len(vertices) + torch.arange(len(_v_i)).view(-1, n)
            vertices = torch.cat(
                (vertices, _v_i.matmul(plane.basis) - plane.origin.matmul(plane.basis)))
            normals = torch.cat(
                (normals, norm(_n_i.matmul(plane.basis), axis=1)))
            colours = torch.cat((colours, _c_i))

            # TODO: OPTIMIZE! THERE IS NO WAY THIS MANY TRIANGLES ARE NEEDED
            for i, j, k in combinations(range(n), 3):
                triangles = torch.cat((triangles, new_tris[:, (i, j, k)]))
                triangles = torch.cat((triangles, new_tris[:, (i, k, j)]))

        return Geometry3(vertices, normals, colours, triangles)


class Simplex4(Mesh4):
    def __init__(self, vertices, colours):
        assert len(vertices) == 5 and len(colours) == 5
        vertices = tensor(vertices, dtype=torch.float)
        center = vertices.mean(axis=0, keepdim=True)
        normals = vertices - center
        normals = norm(normals, axis=1)
        tetrahedra = list(combinations(range(5), 4))
        super().__init__(vertices, normals, colours, tetrahedra)


class Floor4(Mesh4):
    def __init__(self, size, height=0, sigma=1, colour=WHITE):
        vertices, normals, colours, tetrahedra = [], [], [], []

        for x in range(0, size[0]):
            for y in range(0, size[1]):
                for w in range(0, size[2]):
                    z = np.random.normal(height, sigma)
                    vertices.append((x, y, z, w))

        for i in range(0, size[0] - 1):
            for j in range(0, size[1] - 1):
                for k in range(0, size[2] - 1):
                    if (i + j + k) % 2:
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i+1, i, i), (j, j, j+1, j), (k, k, k, k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i+1, i+1, i, i), (j+1, j, j+1, j), (k+1, k, k, k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i+1, i, i+1, i+1), (j, j+1, j+1, j+1), (k, k, k, k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i+1, i+1, i+1), (j, j, j, j+1), (k+1, k, k+1, k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i, i, i+1), (j+1, j, j+1, j+1), (k, k+1, k+1, k+1)), size)))
                    else:
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i+1, i+1, i+1), (j, j, j+1, j), (k, k, k, k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i, i+1, i), (j, j+1, j+1, j+1), (k, k, k, k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i, i+1, i), (j, j, j, j+1), (k, k+1, k+1, k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i+1, i+1, i), (j, j+1, j, j+1), (k, k, k+1, k+1)), size)))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i+1, i+1, i+1, i), (j+1, j+1, j, j+1), (k+1, k, k+1, k+1)), size)))

        super().__init__(vertices, normals, colours, tetrahedra)


class Sphere4(Mesh4):
    def __init__(self, center, radius, colour=WHITE, n=8):
        center = np.array(center)
        vertices, normals, colours, tetrahedra = [], [], [], []

        for theta in np.linspace(0, np.pi, n):
            for phi in np.linspace(0, np.pi, n):
                for omega in np.linspace(0, 2*np.pi, 2*n):
                    x = np.cos(theta)
                    y = np.sin(theta)*np.cos(phi)
                    z = np.sin(theta)*np.sin(phi)*np.cos(omega)
                    w = np.sin(theta)*np.sin(phi)*np.sin(omega)
                    vertices.append(center + radius*np.array((x, y, z, w)))
                    normals.append(np.array((x, y, z, w)))
                    colours.append(colour)

        for i in range(n-1):
            for j in range(n-1):
                for k in range(2*n-1):
                    if (i + j + k) % 2:
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i+1, i, i), (j, j, j+1, j), (k, k, k, k+1)), (n, n, 2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i+1, i+1, i, i), (j+1, j, j+1, j), (k+1, k, k, k+1)), (n, n, 2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i+1, i, i+1, i+1), (j, j+1, j+1, j+1), (k, k, k, k+1)), (n, n, 2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i+1, i+1, i+1), (j, j, j, j+1), (k+1, k, k+1, k+1)), (n, n, 2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i, i, i+1), (j+1, j, j+1, j+1), (k, k+1, k+1, k+1)), (n, n, 2*n))))
                    else:
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i+1, i+1, i+1), (j, j, j+1, j), (k, k, k, k+1)), (n, n, 2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i, i+1, i), (j, j+1, j+1, j+1), (k, k, k, k+1)), (n, n, 2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i, i+1, i), (j, j, j, j+1), (k, k+1, k+1, k+1)), (n, n, 2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i, i+1, i+1, i), (j, j+1, j, j+1), (k, k, k+1, k+1)), (n, n, 2*n))))
                        tetrahedra.append(list(np.ravel_multi_index(
                            ((i+1, i+1, i+1, i), (j+1, j+1, j, j+1), (k+1, k, k+1, k+1)), (n, n, 2*n))))

        super().__init__(vertices, normals, colours, tetrahedra)
