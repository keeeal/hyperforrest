from __future__ import annotations
from typing import TYPE_CHECKING

from panda3d.core import AmbientLight, DirectionalLight, Vec3, Vec4

if TYPE_CHECKING:
    from main import Game


def setup_light(game: Game) -> None:
    ambientLight = AmbientLight("ambient_light")
    ambientLight.set_color(Vec4(0.3, 0.3, 0.3, 1))
    game.render.set_light(game.render.attach_new_node(ambientLight))

    directionalLight = DirectionalLight("directional_light")
    directionalLight.set_direction(Vec3(-5, -5, -5))
    directionalLight.set_color(Vec4(1, 1, 1, 1))
    directionalLight.set_specular_color(Vec4(1, 1, 1, 1))
    game.render.set_light(game.render.attach_new_node(directionalLight))
