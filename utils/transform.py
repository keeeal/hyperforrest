from panda3d.core import GeomVertexRewriter, Vec4

from utils.math import rotmat


class Union:
    def __call__(self, *meshes):
        return


class Scale:
    """
    Scale a mesh along each axis.

    Args:
        v :: array_like (4,)
            A vector of scale factors applied to each axis.
    """

    def __init__(self, v):
        self.v = Vec4(v)

    def __call__(self, mesh):
        vertex_writer = GeomVertexRewriter(mesh.data, "vertex")
        while not vertex_writer.isAtEnd():
            vertex = vertex_writer.getData4()
            vertex_writer.setData4(
                (
                    vertex.x * self.v.x,
                    vertex.y * self.v.y,
                    vertex.z * self.v.z,
                    vertex.w * self.v.w,
                )
            )

        return mesh


class Translate:
    """
    Move a mesh along the specified vector.

    Args:
        v :: array_like (4,)
            The vector displacement applied to the mesh.
    """

    def __init__(self, v):
        self.v = Vec4(v)

    def __call__(self, mesh):
        vertex_writer = GeomVertexRewriter(mesh.data, "vertex")
        while not vertex_writer.isAtEnd():
            vertex = vertex_writer.getData4()
            vertex_writer.setData4(vertex + self.v)

        return mesh


class Rotate:
    """
    Apply a simple rotation in 4D.

    Args:
        theta :: float
            The rotation amount in radians.
        axis_1 :: int
            The index of the first axis in the plane of rotation.
        axis_1 :: int
            The index of the second axis in the plane of rotation.
    """

    def __init__(self, theta, axis_1, axis_2):
        self.m = rotmat(theta, axis_1, axis_2)

    def __call__(self, mesh):
        vertex_writer = GeomVertexRewriter(mesh.data, "vertex")
        while not vertex_writer.isAtEnd():
            vertex = vertex_writer.getData4()
            vertex_writer.setData4(self.m.xform(vertex))

        return mesh
