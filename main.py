import os, sys
from pathlib import Path
from configparser import ConfigParser
from dataclasses import dataclass

import esper

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    DirectionalLight,
    PythonTask,
    Shader,
    Vec3,
)
from panda3d.core import load_prc_file

from utils.shapes import *
from utils.colour import *
from utils.math import *
from utils.transform import *


class Game(ShowBase):
    def __init__(self) -> None:
        load_prc_file(Path("config") / "config.prc")
        super().__init__()

        self.world = esper.World()
        self.view_plane = ViewPlane()
        self.inputs = {}

        setup_controls(self, Path("config") / "controls.ini")

        setup_camera(self)

        setup_shader(self)

        setup_lighting(self)

        setup_test_sphere(self)

        self.task_mgr.add(
            update_world, "update_world", extraArgs=(self,), appendTask=True
        )

        self.task_mgr.add(
            update_view_plane, "update_view_plane", extraArgs=(self,), appendTask=True
        )

        self.task_mgr.add(
            update_exit, "update_exit", extraArgs=(self,), appendTask=True
        )


# Components


class Position(Vec4):
    pass


@dataclass
class ViewPlane:
    origin: Vec4 = Vec4()
    basis: Mat4 = Mat4()


# Systems


def update_exit(game: Game, task: PythonTask) -> int:
    if game.inputs["exit"]:
        sys.exit()

    return task.cont


def update_world(game: Game, task: PythonTask) -> int:
    game.world.process()
    return task.cont


def update_view_plane(game: Game, task: PythonTask) -> int:
    if game.inputs["turn_ana"] or game.inputs["turn_kata"]:
        if game.inputs["turn_ana"]:
            game.view_plane.basis = rotmat(+0.05) * game.view_plane.basis

        if game.inputs["turn_kata"]:
            game.view_plane.basis = rotmat(-0.05) * game.view_plane.basis

        game.render.set_shader_input("plane_origin", game.view_plane.origin)
        game.render.set_shader_input("plane_basis", game.view_plane.basis)

    return task.cont


# Start-up systems


def setup_controls(game: Game, path) -> None:
    config = ConfigParser()
    config.read(path)
    controls = config["controls"]

    def set_inputs(key: str, value: bool) -> None:
        game.inputs[controls[key]] = value

    for key, action in controls.items():
        game.inputs[action] = False
        game.accept(key, set_inputs, [key, True])
        game.accept(key + "-up", set_inputs, [key, False])


def setup_test_sphere(game: Game) -> None:
    sphere = Translate((0, 0, 0, 3))(
        Sphere4(2, colours=[(240 / 255, 135 / 255, 99 / 255, 1)])
    )
    game.render.attachNewNode(sphere.node)


def setup_shader(game: Game) -> None:
    my_shader = Shader.load(
        Shader.SL_GLSL,
        vertex=os.path.join("slicer", "slicer.vert"),
        geometry=os.path.join("slicer", "slicer.geom"),
        fragment=os.path.join("slicer", "slicer.frag"),
    )

    game.render.set_shader(my_shader)
    game.render.setTwoSided(True)
    game.render.set_shader_input("plane_origin", game.view_plane.origin)
    game.render.set_shader_input("plane_basis", game.view_plane.basis)


def setup_camera(game: Game) -> None:
    # game.setBackgroundColor(94/255, 39/255, 80/255)
    game.render.setAntialias(AntialiasAttrib.MAuto)
    game.camera.set_pos(10, 10, 10)
    game.camera.look_at(0, 0, 0)
    game.disable_mouse()


def setup_lighting(game: Game) -> None:
    ambientLight = AmbientLight("ambientLight")
    ambientLight.setColor(Vec4(0.3, 0.3, 0.3, 1))
    game.render.setLight(game.render.attachNewNode(ambientLight))

    directionalLight = DirectionalLight("directionalLight")
    directionalLight.setDirection(Vec3(-5, -5, -5))
    directionalLight.setColor(Vec4(1, 1, 1, 1))
    directionalLight.setSpecularColor(Vec4(1, 1, 1, 1))
    game.render.setLight(game.render.attachNewNode(directionalLight))


if __name__ == "__main__":
    game = Game()
    game.run()
