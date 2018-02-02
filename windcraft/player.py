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
class Player(pygame.sprite.Sprite):
    """This class represents the cursor."""

    def __init__(self):
        """ Constructor for Player."""
        # Call the parent's constructor
        super().__init__()

        self.colors = colors.Colors()
        self.width = 1
        self.height = 1
        self.turbine_path = os.path.join(os.path.dirname(__file__),
                                         'art',
                                         "turbine_24x24_00.png")
        self.invalid_path = os.path.join(os.path.dirname(__file__),
                                         'art',
                                         "invalid_turbine_24x24_00.png")
        self.image = pygame.Surface([self.width, self.height])
        self.rect = self.image.get_rect()

        mouse = pygame.Surface([self.width, self.height])
        self.mouse_rect = mouse.get_rect()

        self.screenheight = pygame.display.get_surface().get_height()
        self.screenwidth = pygame.display.get_surface().get_width()

    def set_turbine(self, farm):
        """Define the wind turbine graphics.

        :param farm: wind farm
        :type farm: :class:`Farm`
        """
        self.size = int(farm.rect.height * 0.2)
        self.margin = self.size * 0.2
        self.turbine_image = pygame.image.load(self.turbine_path).convert()
        self.turbine_image = pygame.transform.scale(
            self.turbine_image, (self.size, self.size))
        self.invalid_turbine_image = pygame.image.load(
            self.invalid_path).convert()
        self.invalid_turbine_image = pygame.transform.scale(
            self.invalid_turbine_image, (self.size, self.size))
        self.turbine_image.set_colorkey(self.colors.black)
        self.rect = self.image.get_rect()

    def update(self, farm, turbines):
        """Update the player position.

        :param farm: wind farm
        :type farm: :class:`Farm`
        :param turbines: turbines in farm
        :type turbines: list
        """
        pos = pygame.mouse.get_pos()
        self.rect.centerx = pos[0]
        self.rect.centery = pos[1]
        self.mouse_rect.centerx = pos[0]
        self.mouse_rect.centery = pos[1]

        # Player does not go off screen
        if self.rect.centerx > self.screenwidth:
            self.rect.centerx = self.screenwidth
        if self.rect.centery > self.screenheight:
            self.rect.centery = self.screenheight

        # Switch mouse representation if we are in the farm
        if not self.mouse_rect.colliderect(farm.inner_rect):
            self.image = pygame.Surface([self.width, self.height])
        else:
            self.image = self.turbine_image
            self.rect = self.image.get_rect()
            self.rect.centerx = pos[0]
            self.rect.centery = pos[1]
            for turbine in turbines:
                if self.rect.colliderect(turbine.rect.inflate(
                        turbine.margin,
                        turbine.margin)):
                    self.image = self.invalid_turbine_image
                    break

        self.image.set_colorkey(self.colors.black)
        self.rect = self.image.get_rect()
        self.rect.centerx = pos[0]
        self.rect.centery = pos[1]
