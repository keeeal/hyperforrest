
import sys, random
import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
from panda3d.core import DirectionalLight

from r4 import *
from colour import *

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

        window = WindowProperties()
        window.setTitle('hyperforrest')
        window.setSize(1920, 1080)
        self.win.requestProperties(window)

        def randPoint4():
            return Point4(*(2*random.random()-1 for i in range(4)))

        self.shapes4 = []

        # self.shapes4.append(
        #     Simplex4((
        #         Vertex4(Point4(0,0,0,0), colour=black),
        #         Vertex4(Point4(0,0,0,1), colour=white),
        #         Vertex4(Point4(0,0,1,0), colour=red),
        #         Vertex4(Point4(0,1,0,0), colour=green),
        #         Vertex4(Point4(1,0,0,0), colour=blue),
        #     ))
        # )

        # self.shapes4.append(
        #     Floor4((10,10,10), 0, .1, white),
        # )

        self.shapes4.append(
            Sphere4(None, 1, n=8)
        )

        self.nodepaths = []

        self.view = Plane4(
            origin=Point4( 0, 0, 0, 0),
            normal=Vec4( 1, 0, 0, 1).normalized(),
            basis= [
                Vec4( 1, 0, 0,-1).normalized(),
                Vec4( 0, 1, 0, 0).normalized(),
                Vec4( 0, 0, 1, 0).normalized(),
            ],
        )
        self.set_view(self.view)

        directionalLight = DirectionalLight('directionalLight')
        directionalLight.setColor((0.9, 0.9, 0.9, 1))
        directionalLightNP = render.attachNewNode(directionalLight)
        directionalLightNP.setHpr(180, -70, 0)
        render.setLight(directionalLightNP)

        self.keys = {
            'q':False, 'w':False, 'e':False, 'a':False, 's':False, 'd':False,
            'arrow_up':False, 'arrow_down':False, 'arrow_left':False, 'arrow_right':False,
            'escape':False, 'control':False,
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
        if theta is not None: self.camera_theta = theta
        if phi is not None: self.camera_phi = phi
        if radius is not None: self.camera_radius = radius
        self.camera.set_pos(
            self.camera_radius*np.sin(self.camera_theta)*np.cos(self.camera_phi),
            self.camera_radius*np.sin(self.camera_theta)*np.sin(self.camera_phi),
            self.camera_radius*np.cos(self.camera_theta),
        )
        self.camera.look_at(0, 0, 0)

    def set_view(self, view=None):
        if view: self.view = view

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
            self.view.normal = Vec4(*rotmat(+.05).dot(self.view.normal)).normalized()
            self.view.basis[0] = Vec4(*rotmat(+.05).dot(self.view.basis[0])).normalized()
            self.set_view()
        if self.keys['d']:
            self.view.normal = Vec4(*rotmat(-.05).dot(self.view.normal)).normalized()
            self.view.basis[0] = Vec4(*rotmat(-.05).dot(self.view.basis[0])).normalized()
            self.set_view()

        return task.cont

def main():
    game = Game()
    game.run()

if __name__ == '__main__':
    main()
