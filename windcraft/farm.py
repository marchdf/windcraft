# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

# ========================================================================
#
# Imports
#
# ========================================================================
import pygame

import windcraft.colors as colors
import windcraft.fonts as fonts


# ========================================================================
#
# Class definitions
#
# ========================================================================
class Farm(pygame.sprite.Sprite):
    """This represents the wind farm."""

    def __init__(self):
        """Constructor for Farm."""
        # Call the parent's constructor
        super().__init__()

        self.colors = colors.Colors()
        self.fonts = fonts.Fonts()
        self.screenheight = pygame.display.get_surface().get_height()
        self.screenwidth = pygame.display.get_surface().get_width()
        self.width = int(self.screenwidth * 0.6)
        self.height = int(self.screenheight * 0.6)

        self.image = pygame.Surface([self.width, self.height])
        self.image.set_colorkey(self.colors.black)

        self.rect = self.image.get_rect()
        self.rect.centerx = self.screenwidth * 0.5
        self.rect.centery = self.screenheight * 0.5

        self.line_width = 5
        self.half_line_width = self.line_width * 0.5

        self.turbine_ratio = 0.2
        self.turbine_size = int(self.rect.height * self.turbine_ratio)

        self.bulb_ratio = 0.1
        self.bulb_size = int(self.rect.height * self.bulb_ratio)

        # Inner farm where turbines are allowed
        self.inner_rect = self.rect.copy()
        self.inner_rect.width = self.rect.width - self.turbine_size
        self.inner_rect.height = self.rect.height - self.turbine_size
        self.inner_rect.centerx = self.rect.centerx
        self.inner_rect.centery = self.rect.centery

    def update(self, canvas):
        """Update the display field in the farm.

        :param canvas: raw canvas of RGB values
        :type canvas: list
        """
        self.image = pygame.image.fromstring(canvas, (self.width, self.height), "ARGB")

    def display(self, screen):
        """Display the farm boundaries.

        :param screen: pygame screen
        :type screen: screen
        """

        # Top line
        pygame.draw.rect(
            screen,
            self.colors.nrel,
            [
                self.rect.x,
                self.rect.y - self.half_line_width,
                self.rect.width,
                self.line_width,
            ],
        )
        # bottom line
        pygame.draw.rect(
            screen,
            self.colors.nrel,
            [
                self.rect.x - self.half_line_width,
                self.rect.bottom - self.half_line_width,
                self.rect.width + self.half_line_width,
                self.line_width,
            ],
        )
        # left line
        pygame.draw.rect(
            screen,
            self.colors.nrel,
            [
                self.rect.x - self.half_line_width,
                self.rect.y - self.half_line_width,
                self.line_width,
                self.rect.height + self.line_width,
            ],
        )
        # right line
        pygame.draw.rect(
            screen,
            self.colors.nrel,
            [
                self.rect.right - self.half_line_width,
                self.rect.y - self.half_line_width,
                self.line_width,
                self.rect.height + self.line_width,
            ],
        )
