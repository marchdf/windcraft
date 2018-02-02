# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.
# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import matplotlib
import colorcet as cc
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt


# ========================================================================
#
# Class definitions
#
# ========================================================================
class Solver():
    """This represents the solver."""

    def __init__(self, farm_width, farm_height):
        """Constructor for Solver.

        :param farm_width: wind farm width
        :type farm_width: int
        :param farm_height: wind farm height
        :type farm_height: int
        """

        # Geometry
        npoints = 100
        self.height = 1.0
        self.width = self.height * farm_width / farm_height
        self.n = [int(npoints * self.width), npoints]
        self.dx = self.width / (self.n[0] - 1)
        self.dy = self.height / (self.n[1] - 1)
        self.dxmin = min(self.dx, self.dy)

        # Start from scratch
        self.reset()

        # Plotting
        self.fig = plt.figure(0,
                              figsize=[1, farm_height / farm_width],
                              dpi=farm_width)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.axis('off')
        self.fig.add_axes(self.ax)
        # self.cmap = cc.cm["diverging_bkr_55_10_c35"]
        # self.cmap = cc.cm["fire"]
        # self.cmap = plt.get_cmap("Blues_r")
        self.cmap = cc.cm["kbc"]
        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.renderer = self.canvas.get_renderer()
        self.im = self.ax.imshow(self.umag.T,
                                 interpolation='none',
                                 cmap=self.cmap,
                                 vmin=0.3,
                                 vmax=self.u0 * 1.3,
                                 origin='lower',
                                 aspect='auto')

    def reset(self):
        """Reset the solver."""
        self.u0 = 4.0
        self.v_pert = self.u0 * 0.1
        self.dpdx = 2.0
        self.power = 0.0
        self.time = 0.0
        self.step = 0
        self.cfl = 0.5
        self.dt = self.dx / (3 * self.u0) * self.cfl
        self.niter = 50

        self.u = self.u0 * np.ones((self.n[0] - 1, self.n[1]))
        self.v = np.zeros((self.n[0], self.n[1] - 1))
        self.ubc = np.zeros((self.n[0] + 1, self.n[1] + 2))
        self.vbc = np.zeros((self.n[0] + 2, self.n[1] + 1))
        self.apply_velocity_bc()

        self.p = np.zeros((self.n[0], self.n[1]))
        self.umag = np.zeros((self.n[0], self.n[1]))

    def solve(self, steps, turbines):
        """Solve the incompressible Euler equations.

        :param steps: number of time steps to solve
        :type steps: int
        :param turbines: turbines in domain
        :type steps: list
        """
        self.power = 0
        for step in range(steps):

            # Apply boundary conditions
            self.apply_velocity_bc()

            # Advection update
            self.advection()

            # Solve Poisson equation for pressure
            self.solve_pressure_poisson()

            # Project velocity
            self.project_velocity()

            # Turbines
            power = self.turbine_update(turbines)

            # Pressure gradient, power output, increments
            self.u += self.dt * self.dpdx
            self.power += self.dt * power
            self.time += self.dt
            self.step += 1

        self.power /= steps

    def apply_velocity_bc(self):
        """Apply the boundary conditions on velocity.

        The inlet is a constant flow. The top, bottom and outlet are
        zero gradient.

        """
        # Interior
        self.ubc[1:-1, 1:-1] = self.u
        self.vbc[1:-1, 1:-1] = self.v

        # West (inflow)
        self.ubc[0, 1:-1] = self.u0 * np.ones((1, self.n[1]))
        self.vbc[0, 1:-1] = self.v_pert * np.sin(np.pi * self.time * 10)

        # East (outflow - zero gradient)
        self.ubc[-1, 1:-1] = self.u[-1, :]
        self.vbc[-1, 1:-1] = self.v[-1, :]

        # North (outflow - zero gradient)
        self.ubc[1:-1, 0] = self.u[:, 0]
        self.vbc[1:-1, 0] = self.v[:, 0]

        # South (outflow - zero gradient)
        self.ubc[1:-1, -1] = self.u[:, -1]
        self.vbc[1:-1, -1] = self.v[:, -1]

    def advection(self):
        """Calculate advective term of NS with upwinding."""

        # Smooth transition between centered differencing and upwinding
        gamma = min(1.2 * self.dt * max(max(self.u.min(),
                                            self.u.max(),
                                            key=abs) / self.dx,
                                        max(self.v.min(),
                                            self.v.max(),
                                            key=abs) / self.dy),
                    1.0)

        ua = yavg(self.ubc)
        ud = 0.5 * np.diff(self.ubc, axis=1)
        va = xavg(self.vbc)
        vd = 0.5 * np.diff(self.vbc, axis=0)
        uvx = np.diff(ua * va - gamma * np.fabs(ua) * vd, axis=0) / self.dx
        uvy = np.diff(ua * va - gamma * np.fabs(va) * ud, axis=1) / self.dy

        # Calculate the squared velocity derivatives
        ua = xavg(self.ubc[:, 1:-1])
        ud = 0.5 * np.diff(self.ubc[:, 1:-1], axis=0)
        va = yavg(self.vbc[1:-1, :])
        vd = 0.5 * np.diff(self.vbc[1:-1, :], axis=1)
        u2x = np.diff(ua * ua - gamma * np.fabs(ua) * ud, axis=0) / self.dx
        v2y = np.diff(va * va - gamma * np.fabs(va) * vd, axis=1) / self.dy

        # Update u and v
        self.u -= self.dt * (uvy[1:-1, :] + u2x)
        self.v -= self.dt * (uvx[:, 1:-1] + v2y)

    def project_velocity(self):
        """Project the velocity field."""
        self.u -= np.diff(self.p, axis=0) / self.dx
        self.v -= np.diff(self.p, axis=1) / self.dy

    def solve_pressure_poisson(self):
        """Solve the Poisson equation for pressure."""
        self.apply_velocity_bc()
        rhs = np.diff(self.ubc, axis=0)[:, 1:-1] / self.dx \
            + np.diff(self.vbc, axis=1)[1:-1, :] / self.dy
        for i in range(self.niter):
            self.p = 0.25 * (p_iterate(self.p, self.dxmin)
                             - self.dx * self.dy * rhs)

    def turbine_update(self, turbines):
        """Update the turbines and flow field.

        :param turbines: turbines in farm
        :type turbines: list
        """
        power = 0
        for turbine in turbines:

            # Turbine location and size
            idx = int(turbine.relative_pos[0] * self.n[0])
            idy = int((1 - turbine.relative_pos[1]) * self.n[1])
            radius = int(turbine.radius * self.n[1])

            # Update turbine speed
            turbine.update(np.mean(self.u[idx, idy - radius:idy + radius]))

            # Apply the turbine force
            force = -0.5 * 4. / 3. * 0.75 * turbine.speed**2 / self.dy
            power -= force
            self.u[idx, idy - radius:idy + radius] += self.dt * force

        return power

    def draw_field(self):
        """Make a raw canvas of a field for pygame."""

        self.apply_velocity_bc()
        self.umag = np.sqrt(xavg(self.ubc)[:, 1:-1]**2
                            + yavg(self.vbc)[1:-1, :]**2)

        self.im.set_data(self.umag.T)
        self.canvas.draw()
        self.raw_canvas = self.renderer.tostring_rgb()


def xavg(f):
    """Calculate the forward average in x.

    :param f: data to average
    :type f: array
    :return: averaged data in x
    :rtype: array
    """
    return 0.5 * (f[1:, :] + f[:-1, :])


def yavg(f):
    """Calculate the forward average in y

    :param f: data to average
    :type f: array
    :return: averaged data in y
    :rtype: array
    """
    return 0.5 * (f[:, 1:] + f[:, :-1])


def p_iterate(p, dxinv):
    """Calculate iteration for Poisson solver (with BC).

    :param p: pressure
    :type p: array
    :return: pressure iterate
    :rtype: array
    """

    iterate = np.roll(p, 1, axis=1) \
        + np.roll(p, -1, axis=1) \
        + np.roll(p, 1, axis=0) \
        + np.roll(p, -1, axis=0)

    # Apply non-periodic BC on left/right
    iterate[0, :] -= p[-1, :]
    iterate[-1, :] -= p[0, :]
    return iterate
