# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import pygame


# ========================================================================
#
# Class definitions
#
# ========================================================================
class Fonts:
    """Defines the fonts used in the game."""

    def __init__(self):
        """Constructor for Fonts."""
        self.screenheight = pygame.display.get_surface().get_height()
        self.screenwidth = pygame.display.get_surface().get_width()

        self.font_path = os.path.join(
            os.path.dirname(__file__), "fonts", "PressStart2P.ttf"
        )

        self.types = {
            "small": pygame.font.Font(self.font_path, int(self.screenheight / 50)),
            "medium": pygame.font.Font(self.font_path, int(self.screenheight / 40)),
            "large": pygame.font.Font(self.font_path, int(self.screenheight / 30)),
        }
