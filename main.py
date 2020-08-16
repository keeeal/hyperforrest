
import os, sys, json
from itertools import combinations, product

import torch
import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile
from panda3d.core import Shader, Vec4
from panda3d.core import *

from utils.r4 import *
from utils.colour import *
from utils.math import rotmat
from utils.transform import *

loadPrcFile(os.path.join('config', 'config.prc'))

# pipeline_path = '/home/j/RenderPipeline/'
# sys.path.insert(0, pipeline_path)
# from rpcore import RenderPipeline


class Game(ShowBase):
    def __init__(self):
        super().__init__()

        # self.render_pipeline = RenderPipeline()
        # self.render_pipeline.create(self)
        # self.render_pipeline.daytime_mgr.time = "11:25"

        # self.render_pipeline.set_effect(
        #     render, "scene-effect.yaml", {}, sort=250)


        my_shapes = [
            # Translate((-2.1,-2.1,0,-2.1)) (Terrain4([.4, .4, .4], [10, 10, 10], scale=5, height=1)),
            RandSphere(1, 10000)
        ]

        my_material = Material()
        # my_material.setShininess(5.0)  # Make this material shiny
        # my_material.setBaseColor((0, 0, 1, 1))  # Make this material blue

        my_shader = Shader.load(Shader.SL_GLSL,
            vertex=os.path.join('slicer', 'slicer.vert'),
            fragment=os.path.join('slicer', 'slicer.frag'),
            geometry=os.path.join('slicer', 'slicer.geom'))

        self.view = Plane4(
            Vec4(0, 0, 0, 0),
            Vec4(0, 0, 0, 1),
            Mat4(
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            )
        )

        self.node_paths = [render.attachNewNode(s.node) for s in my_shapes]
        for node_path in self.node_paths:
            node_path.set_shader(my_shader)
            node_path.setTwoSided(True)

            node_path.setMaterial(my_material)

            node_path.set_shader_input('plane_origin', self.view.origin)
            node_path.set_shader_input('plane_normal', self.view.normal)
            node_path.set_shader_input('plane_basis', self.view.basis)

        controls = os.path.join('config', 'controls.json')
        self.load_controls(controls)

        self.set_camera(1, 1, 8)
        self.disable_mouse()
        self.taskMgr.add(self._loop, 'loop')

        # Create some lighting
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor(Vec4(.3, .3, .3, 1))
        self.render.setLight(self.render.attachNewNode(ambientLight))

        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(Vec3(-5, -5, -5))
        directionalLight.setColor(Vec4(1, 1, 1, 1))
        directionalLight.setSpecularColor(Vec4(1, 1, 1, 1))
        self.render.setLight(self.render.attachNewNode(directionalLight))

    def set_key(self, key, value):
        self.keys[key] = value

    def load_controls(self, controls):
        self.keys, self.ctrl = {}, {}
        with open(controls) as f:
            for key, control in json.load(f).items():
                self.keys[key] = False
                self.ctrl[key] = getattr(self, control)
                self.accept(key, self.set_key, [key, True])
                self.accept(key + '-up', self.set_key, [key, False])

    def set_camera(self, theta=None, phi=None, radius=None):
        if theta is not None:
            self.camera_theta = theta
        if phi is not None:
            self.camera_phi = phi
        if radius is not None:
            self.camera_radius = radius
        self.camera.set_pos(
            self.camera_radius * np.sin(self.camera_theta) *
            np.cos(self.camera_phi),
            self.camera_radius * np.sin(self.camera_theta) *
            np.sin(self.camera_phi),
            self.camera_radius * np.cos(self.camera_theta),
        )
        self.camera.look_at(0, 0, 0)

    def turn_ana(self):
        r = rotmat(+.05)
        self.view.normal = r.xform(self.view.normal)
        self.view.basis = r * self.view.basis
        for node_path in self.node_paths:
            node_path.set_shader_input('plane_normal', self.view.normal)
            node_path.set_shader_input('plane_basis', self.view.basis)

    def turn_kata(self):
        r = rotmat(-.05)
        self.view.normal = r.xform(self.view.normal)
        self.view.basis = r * self.view.basis
        for node_path in self.node_paths:
            node_path.set_shader_input('plane_normal', self.view.normal)
            node_path.set_shader_input('plane_basis', self.view.basis)

    def walk_forward(self):
        pass

    def walk_backwards(self):
        pass

    def walk_left(self):
        pass

    def walk_right(self):
        pass

    def camera_up(self):
        self.set_camera(theta=max(self.camera_theta - .1, 1*np.pi/8))

    def camera_down(self):
        self.set_camera(theta=min(self.camera_theta + .1, 7*np.pi/8))

    def camera_left(self):
        self.set_camera(phi=self.camera_phi - .1)

    def camera_right(self):
        self.set_camera(phi=self.camera_phi + .1)

    def close(self):
        sys.exit()

    def hyper(self):
        pass

    def _loop(self, task):
        for key, pressed in self.keys.items():
            if pressed:
                self.ctrl[key]()

        return task.cont


def main():
    game = Game()
    game.run()


if __name__ == '__main__':
    with torch.no_grad():
        main()
