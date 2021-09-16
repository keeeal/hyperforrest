from __future__ import annotations
from typing import TYPE_CHECKING
from random import random

from direct.showbase.DirectObject import DirectObject
from numpy.random import rand
from panda3d.core import PythonTask, Vec4

import esper

from utils.transform import Translate
from utils.shapes import Sphere4

if TYPE_CHECKING:
    from main import Game


class Position(Vec4):
    pass


class World(DirectObject):
    def __init__(self, game: Game) -> None:
        super().__init__()
        self.world = esper.World()

        for translation in (
            ( 20, 0, 0, 0),
            (-20, 0, 0, 0),
            (0,  20, 0, 0),
            (0, -20, 0, 0),
            (0, 0,  20, 0),
            (0, 0, -20, 0),
        ):
            sphere = Translate(translation)(
                Sphere4(2, colours=[tuple(random() for _ in range(3)) + (1,)])
            )
            game.render.attach_new_node(sphere.node)

    def update(self, task: PythonTask) -> int:
        self.world.process()
        return task.cont
