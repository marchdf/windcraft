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
class Turbine(pygame.sprite.Sprite):
    """This represents the turbine."""

    def __init__(self, farm):
        """ Constructor for Turbine.

        :param farm: wind farm
        :type farm: :class:`Farm`
        """

        # Call the parent's constructor
        super().__init__()

        self.colors = colors.Colors()
        self.size = farm.turbine_size
        self.margin = farm.turbine_size * 0.4
        self.rotation_counter = 0
        self.speed = 0.0
        self.alpha = 0.5
        self.radius = 0.0

        # Turbine images
        art_dir = os.path.join(os.path.dirname(__file__),
                               'art',)
        self.filenames = [os.path.join(art_dir, fname)
                          for fname in ["turbine_24x24_00.png",
                                        "turbine_24x24_01.png",
                                        "turbine_24x24_02.png",
                                        "turbine_24x24_03.png"]]
        self.images = [pygame.image.load(fname).convert()
                       for fname in self.filenames]
        self.images = [pygame.transform.scale(image, (self.size, self.size))
                       for image in self.images]
        for image in self.images:
            image.set_colorkey(self.colors.black)

        self.image = self.images[0]
        self.rect = self.image.get_rect()

    def rotate(self):
        """Pick another image to rotate the turbine"""
        self.rotation_counter = (
            self.rotation_counter + 1) % len(self.filenames)
        loc = self.rect.center
        self.image = self.images[self.rotation_counter]
        self.image.get_rect().center = loc
        self.rect = self.image.get_rect()
        self.rect.center = loc

    def update(self, u):
        """Use local velocity to update the turbine speed.

        :param u: velocity at the turbine
        :type u: float
        """
        self.speed += self.alpha * (u - self.speed)

    def place_turbine(self, pos, turbines, farm):
        """Place turbine in the wind farm.

        :param pos: position to place turbine
        :type pos: list
        :param turbines: current turbines in the farm
        :type turbines: list
        :param farm: wind farm
        :type farm: :class:`Farm`
        """
        mouse = pygame.Surface([1, 1])
        mouse = mouse.get_rect()
        mouse.centerx = pos[0]
        mouse.centery = pos[1]

        if not mouse.colliderect(farm.inner_rect):
            return False
        else:
            # Check for turbine-turbine collision
            collision = False
            turbine = self.images[0].get_rect()
            turbine.centerx = mouse.centerx
            turbine.centery = mouse.centery
            for other_turbine in turbines:
                if turbine.colliderect(
                    other_turbine.rect.inflate(
                        self.margin,
                        self.margin)):
                    return False

            if not collision:
                self.rect.centerx = mouse.centerx
                self.rect.centery = mouse.centery
                self.relative_pos = [
                    (self.rect.centerx - farm.rect.x) / farm.width,
                    (self.rect.centery - farm.rect.y) / farm.height]
                self.radius = farm.turbine_ratio * 0.2
                return True
