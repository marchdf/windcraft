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
import windcraft.colors as colors


# ========================================================================
#
# Class definitions
#
# ========================================================================
class Logo(pygame.sprite.Sprite):
    """This class displays the logo."""

    def __init__(self):
        """ Constructor for Logo."""
        # Call the parent's constructor
        super().__init__()

        self.colors = colors.Colors()
        self.xscale = int(0.25 * pygame.display.get_surface().get_width())
        self.yscale = int(715 / 2500 * self.xscale)
        self.filename = os.path.join(os.path.dirname(__file__),
                                     'art',
                                     "blank_logo.jpg")
        self.image = pygame.image.load(self.filename).convert()
        self.image = pygame.transform.scale(self.image,
                                            (self.xscale, self.yscale))
        self.image.set_colorkey(self.colors.black)
        self.rect = self.image.get_rect()
        self.rect.left = 0.02 * pygame.display.get_surface().get_width()
        self.rect.bottom = 0.98 * pygame.display.get_surface().get_height()
