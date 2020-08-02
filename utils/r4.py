
from itertools import combinations

import torch
import numpy as np

from utils.r3 import Geometry3
from utils.colour import WHITE
from utils.math import *


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
        self.origin = tensor(origin, dtype=torch.float)
        self.normal = norm(tensor(normal, dtype=torch.float))
        self.basis = norm(tensor(basis, dtype=torch.float))

    def to(self, device):
        return Plane4(
            self.origin.to(device),
            self.normal.to(device),
            self.basis.to(device),
        )


class Mesh4:
    '''
    Base class for meshes in R4.

    Args:
        vertices :: array_like (N,4)
            A list of points in the mesh.
        normals :: array_like (N,4)
            A list of vectors perpendicular to the mesh at each vertex.
        colours :: array_like (N,4)
            A list of colours at each vertex.
        tetrahedra :: array_like (M,4)
            A list of indices that group vertices into 4-simplices.
    '''

    def __init__(self, vertices, normals, colours, tetrahedra):
        super().__init__()
        self.vertices = tensor(vertices, dtype=torch.float)
        self.normals = tensor(normals, dtype=torch.float)
        self.colours = tensor(colours, dtype=torch.float)
        self.tetrahedra = tensor(tetrahedra, dtype=torch.long)
        self.device = None

        # The average location and radial extent of each tetrahedron is stored
        # to make slicing more efficient.
        if len(tetrahedra):
            self.means = self.vertices[self.tetrahedra].mean(axis=1, keepdims=True)
            self.radii = self.vertices[self.tetrahedra] - self.means
            self.radii = length(self.radii, axis=2).max(axis=1)[0]
            self.means = self.means.squeeze(1)
        else:
            self.radii, self.means = tensor([]), tensor([])

    def __len__(self):
        return len(self.vertices)

    def to(self, device):
        self.device = device
        self.vertices = self.vertices.to(device)
        self.normals = self.normals.to(device)
        self.colours = self.colours.to(device)
        self.tetrahedra = self.tetrahedra.to(device)
        self.radii = self.radii.to(device)
        self.means = self.means.to(device)
        return self

    def slice(self, plane):

        # calculate how far from the plane each point is
        m_dot_n = self.means.matmul(plane.normal)
        q_dot_n = plane.origin.matmul(plane.normal)

        # only consider tetrahedra near the plane
        nearby = (m_dot_n - q_dot_n).abs() < self.radii
        nearby = self.tetrahedra[nearby]

        # a and b are each pair of points in each tetrahedron
        # BUG: each point must be in a and b at least once
        a = nearby[:, (0, 0, 0, 1, 1, 2)]
        b = nearby[:, (1, 2, 3, 2, 3, 3)]

        # calculate f: the fraction of the line connecting a and b that lies
        # to one side of the plane
        v_a_dot_n = self.vertices[a].matmul(plane.normal)
        v_b_dot_n = self.vertices[b].matmul(plane.normal)
        f = (q_dot_n - v_a_dot_n)/(v_b_dot_n - v_a_dot_n)
        f[v_a_dot_n == q_dot_n] = 0
        # f[v_b_dot_n == q_dot_n] = 1

        # from f, determine whether an intersection has occured and, if so,
        # how many times each tetrahedron has been intersected
        intersection = (0 <= f) * (f < 1)
        intersection_count = intersection.sum(axis=1)

        # prepare 3D data structures
        # vertices = torch.empty((0, 3), dtype=torch.float, device=self.vertices.device)
        # normals = torch.empty((0, 3), dtype=torch.float, device=self.normals.device)
        # colours = torch.empty((0, 4), dtype=torch.float, device=self.colours.device)
        # triangles = torch.empty((0, 3), dtype=torch.long, device=self.tetrahedra.device)
        vertices, normals, colours, triangles = [], [], [], []

        # if the edges of a tetrahedron are intersected either 3 or 4 times,
        # it is visible
        for n in 3, 4:
            n_verts = sum(len(v) for v in vertices)

            mask = intersection_count == n
            _a, _b, _f = a[mask], b[mask], f[mask]
            _i = intersection[mask]
            _f_i = _f[_i].unsqueeze(1)

            # calculate 4-vertices of the triangles
            _v_a_i = self.vertices[_a][_i]
            _v_b_i = self.vertices[_b][_i]
            _v_i = _v_a_i + _f_i*(_v_b_i - _v_a_i)
            vertices.append(_v_i)

            # calculate 4-normals of the triangles
            _n_a_i = self.normals[_a][_i]
            _n_b_i = self.normals[_b][_i]
            _n_i = _n_a_i + _f_i*(_n_b_i - _n_a_i)
            normals.append(_n_i)

            # calculate colours of the triangles
            _c_a_i = self.colours[_a][_i]
            _c_b_i = self.colours[_b][_i]
            _c_i = _c_a_i + _f_i*(_c_b_i - _c_a_i)
            colours.append(_c_i)

            _tris = n_verts + torch.arange(len(_v_i)).view(-1, n)
            for i, j, k in combinations(range(n), 3):
                triangles.append(_tris[:, (i, j, k)])
                triangles.append(_tris[:, (i, k, j)])

        vertices = torch.cat(vertices).matmul(plane.basis) - plane.origin.matmul(plane.basis)
        normals = norm(torch.cat(normals).matmul(plane.basis), axis=1)
        # normals = torch.cat(normals).matmul(plane.basis)
        colours = torch.cat(colours)
        triangles = torch.cat(triangles)

        return Geometry3(vertices, normals, colours, triangles)


class Simplex4(Mesh4):
    def __init__(self, vertices, colours=None):
        assert len(vertices) == 5
        if len(colours) < 5:
            colours = 5*[colours if colours else WHITE]
        colours = tensor(colours)

        vertices = tensor(vertices, dtype=torch.float)
        center = vertices.mean(axis=0, keepdim=True)
        combos = [torch.stack(i) for i in combinations(vertices, 4)]

        normals = []
        combos.reverse()
        for vertex, combo in zip(vertices, combos):

            normal = []
            combo = combo - combo[0]
            for n in range(4):
                cols = list(range(4))
                cols.remove(n)
                det = torch.det(combo[1:, cols])
                if n % 2: det *= -1
                normal.append(det)

                # BUG: normal should be dependent on the location of the
                # the remaining point

            normals += 4*[torch.stack(normal)]

        combos.reverse()
        normals.reverse()

        vertices = torch.cat(combos)
        normals = norm(torch.stack(normals), axis=1)
        colours = torch.cat(4*[colours])
        tetrahedra = torch.arange(20).view(5, 4)

        print(vertices)
        print(normals)
        # print(colours)
        # print(tetrahedra)

        super().__init__(vertices, normals, colours, tetrahedra)


class Floor4(Mesh4):
    def __init__(self, size, height=0, sigma=1, colour=WHITE):
        vertices, normals, colours, tetrahedra = [], [], [], []

        for x in range(0, size[0]):
            for y in range(0, size[1]):
                for w in range(0, size[2]):
                    z = np.random.normal(height, sigma)
                    vertices.append((x, y, z, w))
                    normals.append((0, 0, 1, 0))
                    colours.append(colour)

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
    def __init__(self, center, radius, colours=None, n=8):
        center = np.array(center)
        vertices, normals, tetrahedra = [], [], []

        if len(colours) < 5:
            colours = 5*[colours if colours else WHITE]
        colours = tensor(colours)

        # TODO: Convert this into vectorized opertaions

        for theta in np.linspace(0, np.pi, n):
            for phi in np.linspace(0, np.pi, n):
                for omega in np.linspace(0, 2*np.pi, 2*n):
                    x = np.cos(theta)
                    y = np.sin(theta)*np.cos(phi)
                    z = np.sin(theta)*np.sin(phi)*np.cos(omega)
                    w = np.sin(theta)*np.sin(phi)*np.sin(omega)
                    vertices.append(center + radius*np.array((x, y, z, w)))
                    normals.append((x, y, z, w))

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


class QuickSphere4(Mesh4):
    def __init__(self, center, radius, colour=None, n=8):
        self.center = tensor(center)
        self.radius = radius
        if colour:
            self.colour = colour
        else:
            self.colour = WHITE
        self.n = n
        super().__init__([], [], [], [])

    def to(self, device):
        super().to(device)
        self.center = self.center.to(device)
        return self

    def slice(self, plane):
        distance = self.center.matmul(plane.normal).item()
        radius = np.sqrt(self.radius**2 - distance**2)

        if not radius > 0: return

        n = int(radius*self.n/self.radius)
        theta = torch.linspace(0, np.pi, n).to(self.device)
        phi = torch.linspace(0, 2*np.pi, n).to(self.device)
        theta, phi = torch.meshgrid(theta, phi)

        x = radius*theta.cos()
        y = radius*theta.sin()*phi.cos()
        z = radius*theta.sin()*phi.sin()
        x, y, z = x.flatten(), y.flatten(), z.flatten()

        vertices = torch.stack((x, y, z), axis=1)
        normals  = norm(vertices, axis=1)
        vertices += self.center.matmul(plane.basis)
        colours = len(vertices)*[self.colour]

        t = torch.arange(len(vertices)).view(theta.shape)
        t_1 = t[:-1, :-1].flatten()
        t_2 = t[1:, :-1].flatten()
        t_3 = t[:-1, 1:].flatten()
        t_4 = t[1:, 1:].flatten()

        triangles = torch.cat((
            torch.stack((t_1, t_2, t_3), axis=1),
            torch.stack((t_4, t_3, t_2), axis=1)
        ))

        return Geometry3(vertices, normals, colours, triangles)


class Cube4(Mesh4):
    def __init__(self, vertices, colours=None):
        assert len(vertices) == 16
        if colour:
            assert len(colour) == 16
        else:
            colour = 16*[WHITE]

        keys = list(product(*4*[(0, 1)]))

        for key in keys:
            if not sum(key) % 2:
                neighbours = []
                for i in range(4):
                    neighbours.append(list(key))
                    neighbours[-1][i] = int(not neighbours[-1][i])





        super().__init__(vertices, normals, colours, tetrahedra)
