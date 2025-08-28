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
class Arrow(pygame.sprite.Sprite):
    """This class displays the wind direction."""

    def __init__(self):
        """Constructor for Arrow."""
        # Call the parent's constructor
        super().__init__()
        self.colors = colors.Colors()
        self.fonts = fonts.Fonts()
        self.xstart = 0.1 * pygame.display.get_surface().get_width()
        self.ystart = 0.4 * pygame.display.get_surface().get_height()
        self.block_size = int(0.01 * pygame.display.get_surface().get_width())

    def display(self, screen):
        """Display the arrow on the screen.

        :param screen: pygame screen
        :type screen: screen
        """
        text = self.fonts.types["large"].render("Wind", True, self.colors.red)
        textpos = text.get_rect(centerx=self.xstart, centery=self.ystart)
        screen.blit(text, textpos)

        centerline = self.ystart + self.block_size * 4
        start = self.xstart - self.block_size
        self.image = pygame.Surface([7 * self.block_size, 2 * self.block_size])
        self.image.fill(self.colors.red)
        self.rect = self.image.get_rect()
        self.rect.centerx = start
        self.rect.centery = centerline
        pygame.draw.rect(screen, self.colors.red, self.rect, 0)

        start += 3 * self.block_size
        self.image = pygame.Surface([self.block_size, 4 * self.block_size])
        self.image.fill(self.colors.red)
        self.rect = self.image.get_rect()
        self.rect.centerx = start
        self.rect.centery = centerline
        pygame.draw.rect(screen, self.colors.red, self.rect, 0)

        start += self.block_size
        self.image = pygame.Surface([self.block_size, 3 * self.block_size])
        self.image.fill(self.colors.red)
        self.rect = self.image.get_rect()
        self.rect.centerx = start
        self.rect.centery = centerline
        pygame.draw.rect(screen, self.colors.red, self.rect, 0)

        start += self.block_size
        self.image = pygame.Surface([self.block_size, 2 * self.block_size])
        self.image.fill(self.colors.red)
        self.rect = self.image.get_rect()
        self.rect.centerx = start
        self.rect.centery = centerline
        pygame.draw.rect(screen, self.colors.red, self.rect, 0)

        start += self.block_size
        self.image = pygame.Surface([self.block_size, self.block_size])
        self.image.fill(self.colors.red)
        self.rect = self.image.get_rect()
        self.rect.centerx = start
        self.rect.centery = centerline
        pygame.draw.rect(screen, self.colors.red, self.rect, 0)
