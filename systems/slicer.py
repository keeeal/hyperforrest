from __future__ import annotations
from typing import TYPE_CHECKING

from direct.showbase.DirectObject import DirectObject
from panda3d.core import Shader, PythonTask, Vec4, Mat4

from utils.math import rotmat

if TYPE_CHECKING:
    from main import Game


class Slicer(DirectObject):
    def __init__(self, game: Game, path) -> None:
        super().__init__()
        self.render = game.render
        self.origin = Vec4()
        self.basis = Mat4()

        self.actions = {}

        shader = Shader.load(
            Shader.SL_GLSL,
            vertex=path / "slicer.vert",
            geometry=path / "slicer.geom",
            fragment=path / "slicer.frag",
        )

        game.render.set_shader(shader)
        game.render.set_two_sided(True)
        game.render.set_shader_input("plane_origin", self.origin)
        game.render.set_shader_input("plane_basis", self.basis)

        for action in "turn-ana", "turn-kata":
            self.accept(action, self.actions.update, [{action: True}])
            self.accept(action + "-up", self.actions.update, [{action: False}])

    def update(self, task: PythonTask) -> int:
        change = False

        if self.actions.get("turn-ana"):
            self.basis = rotmat(+0.05) * self.basis
            change = True

        if self.actions.get("turn-kata"):
            self.basis = rotmat(-0.05) * self.basis
            change = True

        if change:
            self.render.set_shader_input("plane_origin", self.origin)
            self.render.set_shader_input("plane_basis", self.basis)

        return task.cont
