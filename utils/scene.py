
class Node:
    def __init__(self, mesh=None, parent=None, children=None):
        self.mesh = mesh
        self.parent = None
        self.children = []