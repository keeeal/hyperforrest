
import sys
import random

import torch
import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import DirectionalLight
from panda3d.core import loadPrcFile

from utils.r4 import *
from utils.colour import *
from utils.math import *

loadPrcFile('config/config.prc')


class Game(ShowBase):
    def __init__(self):
        super().__init__()

        self.shapes4 = []

        self.shapes4.append(
            Simplex4(
                # 2*torch.rand(5, 4) - 1,
                torch.eye(5, 4),
                (
                    GREEN,
                    WHITE,
                    BLACK,
                    RED,
                    BLUE,
                )
            )
        )

        self.nodepaths = []

        self.view = Plane4(
            origin=(0, 0, 0, 0),
            normal=(0, 0, 0, 1),
            basis=torch.eye(4, 3),
        )

        self.set_view(self.view)

        # directionalLight = DirectionalLight('directionalLight')
        # directionalLight.setColor((0.9, 0.9, 0.9, 1))
        # directionalLightNP = render.attachNewNode(directionalLight)
        # directionalLightNP.setHpr(180, -70, 0)
        # render.setLight(directionalLightNP)

        self.keys = {
            'q': False, 'w': False, 'e': False, 'a': False, 's': False, 'd': False,
            'arrow_up': False, 'arrow_down': False, 'arrow_left': False, 'arrow_right': False,
            'escape': False, 'control': False,
        }
        for key in self.keys:
            self.accept(key, self.set_key, [key, True])
            self.accept(key + '-up', self.set_key, [key, False])

        self.set_camera(1, 1, 7)
        self.disable_mouse()
        taskMgr.add(self.loop, 'loop')

    def set_key(self, key, value):
        self.keys[key] = value

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

    def set_view(self, view=None):
        if view:
            self.view = view

        for nodepath in self.nodepaths:
            nodepath.remove_node()
        self.nodepaths = []

        for t4 in self.shapes4:
            t3 = t4.slice(self.view)
            if t3:
                node = t3.get_node()
                nodepath = render.attach_new_node(node)
                self.nodepaths.append(nodepath)

    def loop(self, task):
        if self.keys['escape']:
            sys.exit()

        if self.keys['arrow_up']:
            self.set_camera(theta=max(self.camera_theta - .1, 1*np.pi/8))
        if self.keys['arrow_down']:
            self.set_camera(theta=min(self.camera_theta + .1, 7*np.pi/8))
        if self.keys['arrow_left']:
            self.set_camera(phi=self.camera_phi - .1)
        if self.keys['arrow_right']:
            self.set_camera(phi=self.camera_phi + .1)

        if self.keys['a']:
            self.view.normal = norm(rotmat(+.05).matmul(self.view.normal))
            self.view.basis = norm(rotmat(+.05).matmul(self.view.basis))
            self.set_view()
        if self.keys['d']:
            self.view.normal = norm(rotmat(-.05).matmul(self.view.normal))
            self.view.basis = norm(rotmat(-.05).matmul(self.view.basis))
            self.set_view()

        # print(self.view.normal)

        return task.cont


def main():
    with torch.no_grad():
        game = Game()
        game.run()


if __name__ == '__main__':
    main()
