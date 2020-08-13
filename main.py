
import os, sys, json
from itertools import combinations

import torch
import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import DirectionalLight
from panda3d.core import loadPrcFile

# from utils.r4 import *
# from utils.colour import *

loadPrcFile(os.path.join('config', 'config.prc'))


class Simplex4():
    def __init__(self, vertices):
        super.__init__()



class Game(ShowBase):
    def __init__(self):
        super().__init__()

        self.gpu = None
        # if torch.cuda.is_available():
        #     self.gpu = torch.device('cuda')

        self.slice = []

        for i in range(1):
            my_shape = Sphere4().to(self.gpu)

            self.slice.append(my_shape)

            my_node = my_shape.node
            my_node_path = self.render.attachNewNode(my_node)
            my_node_path.setTwoSided(True)

        self.view = Plane4(
            origin=(0, 0, 0, 0),
            normal=(0, 0, 0, 1),
            basis=torch.eye(4, 3),
        ).to(self.gpu)

        self.view_changed = True

        controls = os.path.join('config', 'controls.json')
        self.load_controls(controls)

        self.set_camera(1, 1, 8)
        self.disable_mouse()
        self.taskMgr.add(self._loop, 'loop')
        self.taskMgr.add(self._slice, 'slice')

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
        r = rotmat(+.05).to(self.gpu)
        self.view.normal = norm(r.matmul(self.view.normal))
        self.view.basis = norm(r.matmul(self.view.basis))
        self.view_changed = True

    def turn_kata(self):
        r = rotmat(-.05).to(self.gpu)
        self.view.normal = norm(r.matmul(self.view.normal))
        self.view.basis = norm(r.matmul(self.view.basis))
        self.view_changed = True

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

    def _slice(self, task):
        if self.view_changed:
            for mesh in self.slice:
                mesh.slice(self.view)

        self.view_changed = False
        return task.cont


def main():
    game = Game()
    game.run()


if __name__ == '__main__':
    with torch.no_grad():
        main()
