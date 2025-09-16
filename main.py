#!/usr/bin/env python
"""
Wind Farm Optimization game

Copyright 2018 National Renewable Energy Laboratory.
This software is released under the license detailed
in the file, LICENSE, which is located in the top-level
directory structure.
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import sys

import pygame

import windcraft as wc


# ========================================================================
#
# Function definitions
#
# ========================================================================
def main():
    parser = argparse.ArgumentParser(description="windcraft")
    parser.add_argument(
        "-r",
        "--resolution",
        dest="resolution",
        help="Screen resolution (width x height)",
        type=int,
        nargs=2,
        default=[800, 600],
    )
    parser.add_argument(
        "-f",
        "--fullscreeen",
        dest="fullscreen",
        help="Launch in fullscreen mode",
        action="store_true",
    )
    parser.add_argument(
        "-g",
        "--grid_resolution",
        dest="grid_resolution",
        help="Grid resolution",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-p",
        "--ideal_power",
        dest="ideal_power",
        help="Ideal generated power target in kW",
        type=int,
        default=350,
    )
    args = parser.parse_args()

    # Call this function so the Pygame library can initialize itself
    pygame.init()

    fps = 30

    # Create the screen
    if args.fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode([args.resolution[0], args.resolution[1]])

    # Set the title of the window
    pygame.display.set_caption("windcraft")

    # Enable this to make the mouse disappear when over our window
    pygame.mouse.set_visible(1)

    # Create a surface we can draw on
    pygame.Surface(screen.get_size())

    # Create sprite lists
    allsprites = pygame.sprite.Group()

    # Wind farm
    wind_farm = wc.Farm()
    allsprites.add(wind_farm)

    # Turbines
    turbines = pygame.sprite.Group()
    max_turbines = 100
    rotation_rate = fps / 6
    turbine_list = []

    # Solver
    solver = wc.Solver(wind_farm.width, wind_farm.height, args.grid_resolution)
    solver_steps = int(0.12 / (solver.dt * fps))
    if solver_steps <= 0:
        solver_steps = 30

    # Player
    player = wc.Player()
    player.set_turbine(wind_farm)
    allsprites.add(player)

    # Logo
    logo = wc.Logo()
    allsprites.add(logo)

    # Wind arrow
    arrow = wc.Arrow()

    # Text
    text = wc.Text()

    # Colors
    colors = wc.Colors()

    # Clock to limit speed
    clock = pygame.time.Clock()

    # Loop until close
    done = False

    # Main program loop
    done = False
    count = 0
    while not done:
        clock.tick(fps)

        # Clear the screen
        screen.fill(colors.white)

        # Process the events in the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if len(turbines) < max_turbines:
                    turbine = wc.Turbine(wind_farm)
                    pos = pygame.mouse.get_pos()
                    success = turbine.place_turbine(pos, turbines, wind_farm)
                    if success:
                        turbines.add(turbine)
                        allsprites.add(turbine)
                        turbine_list.append(turbine)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                elif event.key == pygame.K_r:
                    solver.reset()
                    for turbine in turbines:
                        turbine.kill()
                elif event.key == pygame.K_u:
                    if len(turbine_list) > 0:
                        turbine_list[-1].kill()
                        turbine_list = turbine_list[:-1]
                elif event.key == pygame.K_t:
                    print("not implemented yet")

        # Solve flow field and draw
        solver.solve(solver_steps, turbines)
        solver.draw_field()
        wind_farm.update(solver.raw_canvas)

        # Update the player position
        player.update(wind_farm, turbines)

        # Rotate turbines
        if count % rotation_rate == 0:
            for turbine in turbines:
                turbine.rotate()

        # Draw text
        text.display(
            screen, len(turbines), max_turbines, solver.power, args.ideal_power
        )

        # Draw arrow
        arrow.display(screen)

        # Draw sprites
        allsprites.draw(screen)

        # Draw farm boundaries (do this after the sprite update)
        wind_farm.display(screen)

        # Flip the screen and show what we've drawn
        pygame.display.flip()

        count += 1

    pygame.quit()

    return 0


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    sys.exit(main())
