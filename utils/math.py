
import numpy as np
from panda3d.core import Mat4


def length(a, axis=0, keepdims=False):
    print((a**2).sum)
    return (a**2).sum(axis, keepdims=keepdims)**.5


def norm(a, axis=0):
    return a/length(a, axis, keepdims=True)


def rotmat(theta, axis_1=0, axis_2=3):
    m = np.eye(4)
    m[axis_1, axis_1] = np.cos(theta)
    m[axis_1, axis_2] = -np.sin(theta)
    m[axis_2, axis_1] = np.sin(theta)
    m[axis_2, axis_2] = np.cos(theta)
    return Mat4(*m.flatten())
