from __future__ import annotations
from typing import TYPE_CHECKING
from copy import copy

from direct.showbase.DirectObject import DirectObject
from panda3d.core import AntialiasAttrib, PythonTask, WindowProperties, Point2
from math import sin, cos, radians

if TYPE_CHECKING:
    from main import Game


class Camera(DirectObject):
    def __init__(self, game: Game) -> None:
        super().__init__()
        self.camera = game.camera
        self.mouse_watcher = game.mouseWatcherNode
        self.previous_mouse = Point2()
        self.mouse_sensitivity = 50
        self.actions = {}

        # game.setBackgroundColor(94/255, 39/255, 80/255)
        game.render.set_antialias(AntialiasAttrib.MAuto)
        self.camera.set_pos(0, 0, 0)
        self.camera.look_at(1, 0, 0)
        game.disable_mouse()

        props = WindowProperties()
        props.set_cursor_hidden(True)
        props.set_mouse_mode(WindowProperties.M_relative)
        game.win.request_properties(props)

        for action in "walk-forward", "walk-backward", "walk-left", "walk-right":
            self.accept(action, self.actions.update, [{action: True}])
            self.accept(action + "-up", self.actions.update, [{action: False}])

    def update(self, task: PythonTask) -> int:
        if self.mouse_watcher.has_mouse():
            current_mouse = self.mouse_watcher.get_mouse()
            delta = current_mouse - self.previous_mouse

            self.camera.set_h(self.camera.get_h() - self.mouse_sensitivity * delta.x)
            self.camera.set_p(clip(self.camera.get_p() + self.mouse_sensitivity * delta.y, -90, 90))

            self.previous_mouse = copy(current_mouse)

        x = self.camera.get_x()
        y = self.camera.get_y()
        h = self.camera.get_h()

        if self.actions.get('walk-forward'):
            x -= sin(radians(h))
            y += cos(radians(h))

        if self.actions.get('walk-backward'):
            x += sin(radians(h))
            y -= cos(radians(h))

        if self.actions.get('walk-left'):
            x -= cos(radians(h))
            y -= sin(radians(h))

        if self.actions.get('walk-right'):
            x += cos(radians(h))
            y += sin(radians(h))

        self.camera.set_x(x)
        self.camera.set_y(y)

        return task.cont


def clip(x, a, b):
    a, b = sorted((a, b))
    return min(max(a, x), b)
