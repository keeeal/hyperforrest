import os, sys
from pathlib import Path, PosixPath
from configparser import ConfigParser
from dataclasses import dataclass

import esper
import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    DirectionalLight,
    Shader,
    Vec3,
)
from panda3d.core import load_prc_file


from utils.shapes import *
from utils.colour import *
from utils.math import *
from utils.transform import *

# Components


class Position(Vec4):
    pass

class Basis(Mat4):
    pass


# Systems


class ViewPlaneSystem(esper.Processor):
    def __init__(self, view_plane_id: int) -> None:
        super().__init__()
        self.view_plane_id = view_plane_id

    def process(self) -> None:
        basis = self.world.component_for_entity(self.view_plane_id, Basis)





def turn_ana(game):
    r = rotmat(+0.05)
    game.view.basis = r * game.view.basis
    for node_path in game.node_paths:
        node_path.set_shader_input("plane_basis", game.view.basis)


def load_controls(game, path):
    config = ConfigParser()
    config.read(path)
    controls = config["controls"]

    def set_inputs(key: str, value: bool) -> None:
        game.inputs[controls[key]] = value

    for key, action in controls.items():
        game.inputs[action] = False
        game.accept(key, set_inputs, [key, True])
        game.accept(key + "-up", set_inputs, [key, False])


def setup(game):

    my_shapes = []

    my_shapes.append(
        Translate((0, 0, 0, 0))(
            Sphere4(2, colours=[(240 / 255, 135 / 255, 99 / 255, 1)])
        )
    )

    my_shader = Shader.load(
        Shader.SL_GLSL,
        vertex=os.path.join("slicer", "slicer.vert"),
        fragment=os.path.join("slicer", "slicer.frag"),
        geometry=os.path.join("slicer", "slicer.geom"),
    )

    view_plane = game.world.create_entity()
    game.world.add_component(view_plane, Position())
    game.world.add_component(view_plane, Basis())

    game.node_paths = [game.render.attachNewNode(s.node) for s in my_shapes]

    for node_path in game.node_paths:
        node_path.set_shader(my_shader)
        node_path.setTwoSided(True)

        node_path.set_shader_input(
            "plane_origin", game.world.component_for_entity(view, Position)
        )
        node_path.set_shader_input(
            "plane_basis", game.world.component_for_entity(view, Basis)
        )

    # load_controls(game, Path("config") / "controls.ini")
    # game.setBackgroundColor(94/255, 39/255, 80/255)
    game.render.setAntialias(AntialiasAttrib.MAuto)

    game.set_camera(1, 1, 16)
    game.disable_mouse()


def setup_lighting(game):
    # Create some lighting
    ambientLight = AmbientLight("ambientLight")
    ambientLight.setColor(Vec4(0.3, 0.3, 0.3, 1))
    game.render.setLight(game.render.attachNewNode(ambientLight))

    directionalLight = DirectionalLight("directionalLight")
    directionalLight.setDirection(Vec3(-5, -5, -5))
    directionalLight.setColor(Vec4(1, 1, 1, 1))
    directionalLight.setSpecularColor(Vec4(1, 1, 1, 1))
    game.render.setLight(game.render.attachNewNode(directionalLight))


class Game(ShowBase):
    def __init__(self):
        load_prc_file(Path("config") / "config.prc")
        super().__init__()

        # Resources

        self.world = esper.World()
        self.controls = load_controls(Path("config") / "controls.ini")
        self.inputs = {}

        # startup systems

        view_plane = self.world.create_entity()
        self.world.add_component(view_plane, Position())
        self.world.add_component(view_plane, Basis())

        # setup(self)

        setup_lighting(self)

        # systems

        self.world.add_processor(ViewPlaneSystem(view_plane))

        # main loop

        def process_world(task):
            self.world.process()
            return task.cont

        self.task_mgr.add(process_world, "process_world")

        self.task_mgr.add(process_inputs, "process_inputs")

    def set_camera(self, theta=None, phi=None, radius=None):
        if theta is not None:
            self.camera_theta = theta
        if phi is not None:
            self.camera_phi = phi
        if radius is not None:
            self.camera_radius = radius
        self.camera.set_pos(
            self.camera_radius * np.sin(self.camera_theta) * np.cos(self.camera_phi),
            self.camera_radius * np.sin(self.camera_theta) * np.sin(self.camera_phi),
            self.camera_radius * np.cos(self.camera_theta),
        )
        self.camera.look_at(0, 0, 0)

    # def turn_ana(self):
    #     r = rotmat(+0.05)
    #     self.view.basis = r * self.view.basis
    #     for node_path in self.node_paths:
    #         node_path.set_shader_input("plane_basis", self.view.basis)

    # def turn_kata(self):
    #     r = rotmat(-0.05)
    #     self.view.basis = r * self.view.basis
    #     for node_path in self.node_paths:
    #         node_path.set_shader_input("plane_basis", self.view.basis)

    # def walk_forward(self):
    #     pass

    # def walk_backwards(self):
    #     pass

    # def walk_left(self):
    #     pass

    # def walk_right(self):
    #     pass

    # def camera_up(self):
    #     self.set_camera(theta=max(self.camera_theta - 0.1, 1 * np.pi / 8))

    # def camera_down(self):
    #     self.set_camera(theta=min(self.camera_theta + 0.1, 7 * np.pi / 8))

    # def camera_left(self):
    #     self.set_camera(phi=self.camera_phi - 0.1)

    # def camera_right(self):
    #     self.set_camera(phi=self.camera_phi + 0.1)

    # def close(self):
    #     sys.exit()

    # def hyper(self):
    #     pass

    # def loop(self, task):
    #     for key, pressed in self.keys.items():
    #         if pressed:
    #             self.actions[key]()

    #     return task.cont


if __name__ == "__main__":
    game = Game()
    game.run()
