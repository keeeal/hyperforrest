
import random, math

from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point4, Vec4

from shapes4 import *

class Game(ShowBase):
    def __init__(self):
        super().__init__()
        self.view = Plane4(
            Point4(0,0,0,0.5),
            Vec4(0,0,0,1),
        )

        t4 = Tetra4(
            Point4(0,0,0,0),
            Point4(0,0,0,1),
            Point4(0,0,1,0),
            Point4(0,1,0,0),
            Point4(1,0,0,0),
        )

        t3 = t4.slice(self.view)

        nodepath = render.attach_new_node(t3.get_node())

        self.keys = {
            'w':False, 'a':False, 's':False, 'd':False,
            'arrow_up':False, 'arrow_left':False, 'arrow_down':False, 'arrow_right':False,
            'escape':False, 'control':False,
        }
        for key in self.keys:
            self.accept(key, self.set_key, [key, True])
            self.accept(key + '-up', self.set_key, [key, False])

        taskMgr.add(self.loop, 'loop')

        self.camera_theta = self.camera_phi = 0
        self.camera.set_pos(
            math.cos(self.camera_angle),
            math.sin(self.camera_angle),
            5,
        )
        self.camera.look_at(0, 0, 0)
        self.disable_mouse()

    def set_key(self, key, value):
        self.keys[key] = value

    def loop(self, task):
        camera_moved = False

        if self.keys['arrow_left']:
            self.camera_angle += .1
            camera_moved = True
        if self.keys['arrow_right']:
            self.camera_angle -= .1
            camera_moved = True

        if camera_moved:
            self.camera.set_pos(
                math.cos(self.camera_angle),
                math.sin(self.camera_angle),
                5,
            )
            self.camera.look_at(0, 0, 0)

        return task.cont

def main():
    game = Game()
    game.run()

if __name__ == '__main__':
    main()
