
import numpy as np
from panda3d.core import Mat4


def length(a, axis=0, keepdims=False):
    print((a**2).sum)
    return (a**2).sum(axis, keepdims=keepdims)**.5


def norm(a, axis=0):
    return a/length(a, axis, keepdims=True)


def rotmat(theta):
    return Mat4(
        (np.cos(theta), 0, 0, -np.sin(theta)),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (np.sin(theta), 0, 0, np.cos(theta)),
    )
