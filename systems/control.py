from __future__ import annotations
from typing import TYPE_CHECKING
from configparser import ConfigParser

from direct.showbase.DirectObject import DirectObject

if TYPE_CHECKING:
    from main import Game


class Control(DirectObject):
    def __init__(self, game: Game, path) -> None:
        super().__init__()
        config = ConfigParser()
        config.read(path)
        controls = config["controls"]

        # game.messenger.toggle_verbose()

        for key, action in controls.items():
            for suffix in "", "-up":
                # TODO: Pass event arguments to new event
                self.accept(key + suffix, game.messenger.send, [action + suffix])
