
import sys, math, random
import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point4, Vec4
from panda3d.core import WindowProperties

from shapes4 import *

window = WindowProperties()
window.setTitle('hyperforrest')

def rotmat(theta):
    m = np.zeros((4, 4))
    m[0,0] =  np.cos(theta)
    m[0,3] = -np.sin(theta)
    m[3,0] =  np.sin(theta)
    m[3,3] =  np.cos(theta)
    return m

class Game(ShowBase):
    def __init__(self):
        super().__init__()
        self.win.requestProperties(window)

        self.tetra4 = []
        self.tetra4.append(Tetra4(
            Point4(0,0,0,0),
            Point4(0,0,0,1),
            Point4(0,0,1,0),
            Point4(0,1,0,0),
            Point4(1,0,0,0),
        ))

        self.nodepaths = []
        self.view = Plane4(
            Point4(0,0,0,0),
            Vec4(0,0,0,1),
        )
        self.set_view(self.view)

        self.keys = {
            'q':False, 'w':False, 'e':False, 'a':False, 's':False, 'd':False,
            'arrow_up':False, 'arrow_down':False, 'arrow_left':False, 'arrow_right':False,
            'escape':False, 'control':False,
        }
        for key in self.keys:
            self.accept(key, self.set_key, [key, True])
            self.accept(key + '-up', self.set_key, [key, False])

        self.set_camera(1, 1, 5)
        self.disable_mouse()
        taskMgr.add(self.loop, 'loop')

    def set_key(self, key, value):
        self.keys[key] = value

    def set_camera(self, theta=None, phi=None, radius=None):
        if theta: self.camera_theta = theta
        if phi: self.camera_phi = phi
        if radius: self.camera_radius = radius
        self.camera.set_pos(
            self.camera_radius*math.sin(self.camera_theta)*math.cos(self.camera_phi),
            self.camera_radius*math.sin(self.camera_theta)*math.sin(self.camera_phi),
            self.camera_radius*math.cos(self.camera_theta),
        )
        self.camera.look_at(0, 0, 0)

    def set_view(self, view=None):
        if view: self.view = view
        for nodepath in self.nodepaths:
            nodepath.remove_node()
        for t4 in self.tetra4:
            t3 = t4.slice(self.view)
            node = t3.get_node()
            nodepath = render.attach_new_node(node)
            self.nodepaths.append(nodepath)

    def loop(self, task):
        if self.keys['escape']:
            sys.exit()

        if self.keys['arrow_up']:
            self.set_camera(theta=max(self.camera_theta - .1, 1*math.pi/8))
        if self.keys['arrow_down']:
            self.set_camera(theta=min(self.camera_theta + .1, 7*math.pi/8))
        if self.keys['arrow_left']:
            self.set_camera(phi=self.camera_phi - .1)
        if self.keys['arrow_right']:
            self.set_camera(phi=self.camera_phi + .1)

        if self.keys['a']:
            self.view.normal = Vec4(*rotmat(+.1).dot(self.view.normal))
            self.set_view()
        if self.keys['d']:
            self.view.normal = Vec4(*rotmat(-.1).dot(self.view.normal))
            self.set_view()

        return task.cont

def main():
    game = Game()
    game.run()

if __name__ == '__main__':
    main()
