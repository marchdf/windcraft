# Copyright 2025 National Renewable Energy Laboratory. This software
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
import windcraft.fonts as fonts


# ========================================================================
#
# Class definitions
#
# ========================================================================
class Bulbs:
    """This class displays the turbine text."""

    def __init__(self, bulb_size):
        """Constructor for Text."""
        self.colors = colors.Colors()
        self.fonts = fonts.Fonts()
        art_dir = os.path.join(
            os.path.dirname(__file__),
            "art",
        )
        bulb_files = [
            os.path.join(art_dir, fname) for fname in ["bulbon.png", "bulboff.png"]
        ]
        self.bulb_size = bulb_size
        self.bulb_images = [pygame.image.load(fname) for fname in bulb_files]
        self.bulb_images = [
            pygame.transform.scale(image, (self.bulb_size, self.bulb_size))
            for image in self.bulb_images
        ]

    def display(self, screen, power, ideal_power):
        """Display the lit/unlit bulbs on the screen."""
        # Power
        xstart = 0.5 * pygame.display.get_surface().get_width()
        ystart = 0.10 * pygame.display.get_surface().get_height()
        text = self.fonts.types["medium"].render(
            "Power produced : ",
            True,
            self.colors.black,
        )
        textpos = text.get_rect(centerx=xstart, top=ystart)
        screen.blit(text, textpos)

        num_bulbs = 10
        scale_factor = 100
        percent = 100 * power * scale_factor / ideal_power
        xstart = (
            0.5 * pygame.display.get_surface().get_width()
            - num_bulbs // 2 * self.bulb_size
        )
        for i in range(num_bulbs):
            if percent >= (i + 1) * 10:
                bulb_img = self.bulb_images[0]
            else:
                bulb_img = self.bulb_images[1]
            screen.blit(
                bulb_img,
                (xstart + i * self.bulb_size, textpos.top + self.bulb_size * 0.6),
            )
