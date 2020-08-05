
class Transform:
    def to(device):
        pass

class Translate:
    def __init__(self, vector):
        self.vector = tensor(vector)

    def __call__(self, *meshes):
        for mesh in meshes:
            mesh.vertices += self.vector
