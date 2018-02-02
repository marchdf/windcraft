# -*- coding: utf-8 -*-

try:
    __WINDCRAFT_SETUP__
except NameError:
    __WINDCRAFT_SETUP__ = False

if not __WINDCRAFT_SETUP__:
    __all__ = ["Arrow",
               "Colors",
               "Text",
               "Farm",
               "Fonts",
               "Logo",
               "Player",
               "Solver",
               "Turbine"]
    from .arrow import Arrow
    from .colors import Colors
    from .text import Text
    from .fonts import Fonts
    from .farm import Farm
    from .logo import Logo
    from .player import Player
    from .solver import Solver
    from .turbine import Turbine
