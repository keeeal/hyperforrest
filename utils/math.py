import numpy as np
from panda3d.core import Mat4


def length(a, axis=0, keepdims=False):
    return (a ** 2).sum(axis, keepdims=keepdims) ** 0.5


def norm(a, axis=0):
    return a / length(a, axis, keepdims=True)


def rotmat(theta, axis_1=0, axis_2=3):
    m = np.eye(4)
    m[axis_1, axis_1] = np.cos(theta)
    m[axis_1, axis_2] = -np.sin(theta)
    m[axis_2, axis_1] = np.sin(theta)
    m[axis_2, axis_2] = np.cos(theta)
    return Mat4(*m.flatten())


def get_tetra_norms(vertices, tetrahedra, center=None):
    normals = []

    # the normal for each tetrahedron is a perpedicular vector found using
    # the determinant (https://math.stackexchange.com/questions/904172) and
    # then directed away from the center
    for tetra in tetrahedra:
        normals.append(np.empty(4))
        t = vertices[tetra]
        first = t[0]
        t = t - first

        for i in range(4):
            cols = list(range(4))
            cols.remove(i)
            det = np.linalg.det(t[1:, cols])
            normals[-1][i] = -det if i % 2 else det

        if center is not None:
            if normals[-1].dot(first - center) < 0:
                normals[-1] *= -1

    return np.stack(normals)


def halton(n, base=(2, 3)):
    """
    Generate a Halton sequence in the range (0, 1) along each dimension.

    Args:
        n :: iterable
            The indices of the sequence.
        base :: array_like (D,)
            The bases to be used for each dimension, where D is the number of
            dimensions of each sequence element.
    """

    # TODO: Make this more efficient
    def h(i, b):
        f, r = 1, 0
        while i > 0:
            f = f / b
            r = r + f * (i % b)
            i = int(i / b)
        return r

    m = max(base)
    return [[h(i + m, b) for b in base] for i in n]
