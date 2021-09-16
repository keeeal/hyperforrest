import sys
from pathlib import Path

from direct.showbase.ShowBase import ShowBase
from panda3d.core import load_prc_file

from systems.camera import Camera
from systems.control import Control
from systems.light import setup_light
from systems.slicer import Slicer
from systems.world import World


class Game(ShowBase):
    def __init__(self) -> None:
        load_prc_file(Path("config") / "config.prc")
        super().__init__()

        self.control = Control(self, Path("config") / "controls.ini")

        self.slicer = Slicer(self, Path("slicer"))

        self.world = World(self)

        self.camera = Camera(self)

        setup_light(self)

        self.task_mgr.add(self.world.update, "update_world")

        self.task_mgr.add(self.slicer.update, "update_slicer")

        self.task_mgr.add(self.camera.update, "update_camera")

        self.accept("exit", sys.exit)


if __name__ == "__main__":
    game = Game()
    game.run()
