# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.
# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np


# ========================================================================
#
# Class definitions
#
# ========================================================================
class Solver:
    def __init__(self, npts, farm_width, farm_height):
        """Constructor for Solver.
        :param farm_width: wind farm width
        :type farm_width: int
        :param farm_height: wind farm height
        :type farm_height: int
        """
        # some enums
        self.West = 0
        self.East = 1
        self.South = 2
        self.North = 3

        # Geometry
        npoints = npts
        self.height = 1.0
        self.width = self.height * farm_width / farm_height
        self.nnodes = np.array([int(npoints * self.width), int(npoints)])
        self.ncells = self.nnodes - 1
        self.dx = self.width / (self.nnodes[0] - 1)
        self.dy = self.height / (self.nnodes[1] - 1)
        self.dxmin = min(self.dx, self.dy)
        print("ncells:", self.ncells)

        # u is cell centered along y and node centered along x
        # v is node centered along y and cell centered along x
        # add total of 2 ghost points along each direction
        self.ubc = np.zeros((self.ncells[1] + 2, self.nnodes[0] + 2))
        self.vbc = np.zeros((self.nnodes[1] + 2, self.ncells[0] + 2))

        self.presbc = np.zeros(4)  # all neumann boundaries
        self.presbcvals = np.zeros(4)  # dirichlet bc values
        self.presmat = np.zeros(
            (self.ncells[0] * self.ncells[1], self.ncells[0] * self.ncells[1])
        )
        self.presrhs = np.zeros((self.ncells[0] * self.ncells[1]))

        # pressure is cell centered along x and y
        self.p = np.zeros((self.ncells[1] + 2, self.ncells[0] + 2))

        # pressure is cell centered along x and y
        self.div = np.zeros((self.ncells[1] + 2, self.ncells[0] + 2))
        self.divl2norm = 0.0

        # Start from scratch
        self.reset()

    def reset(self):
        """Reset the solver."""
        self.u0 = 4.0
        self.step = 0
        self.cfl = 0.5
        self.dt = min(self.dx, self.dy) / (3 * self.u0) * self.cfl
        self.visc = 0.01

        self.ubc[:, :] = self.u0
        self.apply_velocity_bc()
        self.apply_pres_bc()
        self.computediv()

        self.fillpresmat()

    def solve(self, steps):
        """Solve the incompressible Euler equations."""
        for step in range(steps):
            # Apply boundary conditions
            self.apply_velocity_bc()

            # Advection update
            self.advectdiffuse()
            self.computediv()
            print("velocity divergence before projection:", self.divl2norm)

            # Solve Poisson equation for pressure
            self.solve_pressure_poisson()

            # Project velocity
            self.project_velocity()
            self.computediv()
            print("velocity divergence after projection:", self.divl2norm)
            print("==================================================")

    def apply_velocity_bc(self):
        """Apply the boundary conditions on velocity.
        The inlet is a constant flow.
        The top, bottom are walls
        and outlet are zero gradient.
        """
        # West (inflow)
        self.ubc[:, 0] = self.u0
        self.ubc[:, 1] = self.u0
        self.vbc[:, 0] = 0.0
        self.presbc[self.West] = 0.0  # neumann bc

        # East (outflow - zero gradient)
        self.ubc[:, -1] = self.ubc[:, -2]
        self.vbc[:, -1] = self.vbc[:, -2]
        self.presbc[self.East] = 1.0  # dirchlet bc
        self.presbcvals[self.East] = 0.0

        # South (wall)
        self.ubc[0, 1:-1] = 0.0
        self.vbc[0, 1:-1] = 0.0
        self.vbc[1, 1:-1] = 0.0
        self.presbc[self.South] = 0.0  # neumann bc

        # North (wall)
        self.ubc[-1, 1:-1] = 0.0
        self.vbc[-1, 1:-1] = 0.0
        self.vbc[-2, 1:-1] = 0.0
        self.presbc[self.North] = 0.0  # neumann bc

    def fillpresmat(self):
        self.presrhs[:] = 0.0
        self.presmat[:, :] = 0.0
        dx2 = self.dx**2
        dy2 = self.dy**2
        # compute pressure matrix coefficients only once since they don't change
        # doing this for neumann bc first
        for j in range(self.ncells[1]):
            for i in range(self.ncells[0]):
                myself = j * self.ncells[0] + i
                mywest = j * self.ncells[0] + i - 1
                myeast = j * self.ncells[0] + i + 1
                mysouth = (j - 1) * self.ncells[0] + i
                mynorth = (j + 1) * self.ncells[0] + i

                self.presmat[myself][myself] -= 2.0 / dx2 + 2.0 / dy2

                if (i - 1) < 0:  # west boundary (homogenous neumann)
                    self.presmat[myself][myself] += 1.0 / dx2
                else:
                    self.presmat[myself][mywest] += 1.0 / dx2

                if (i + 1) > self.ncells[0] - 1:  # east boundary (dirichlet)
                    self.presrhs[myself] -= self.presbcvals[self.East] / dx2
                else:
                    self.presmat[myself][myeast] += 1.0 / dx2

                if (j - 1) < 0:  # south boundary (homogenous neumann)
                    self.presmat[myself][myself] += 1.0 / dy2
                else:
                    self.presmat[myself][mysouth] += 1.0 / dy2

                if (j + 1) > self.ncells[1] - 1:  # north boundary (homogenous neumann)
                    self.presmat[myself][myself] += 1.0 / dy2
                else:
                    self.presmat[myself][mynorth] += 1.0 / dy2

        # print self.presmat

    def advectdiffuse(self):
        """Calculate advective term of NS with upwinding."""
        # upwind formulation uv at face = 0.5*(u+mod(u))*vleft+0.5*(u-mod(u))*vright

        ucc = xavg(self.ubc)  # cell center
        vcc = yavg(self.vbc)  # cell center
        vcorner = xavg(self.vbc)
        ucorner = yavg(self.ubc)

        u2l = (ucc + np.fabs(ucc)) * self.ubc[:, :-1]  # left state weighting
        u2r = (ucc - np.fabs(ucc)) * self.ubc[:, 1:]  # right state weighting

        # need to advance only interior points, therefore 1:-1
        u2dx = np.diff(0.5 * (u2l + u2r), axis=1)[1:-1, :] / self.dx

        uvl = (vcorner[1:-1, :] + np.fabs(vcorner[1:-1, :])) * self.ubc[:-1, 1:-1]
        uvr = (vcorner[1:-1, :] - np.fabs(vcorner[1:-1, :])) * self.ubc[1:, 1:-1]
        uvdy = np.diff(0.5 * (uvl + uvr), axis=0) / self.dy

        v2l = (vcc + np.fabs(vcc)) * self.vbc[:-1, :]
        v2r = (vcc - np.fabs(vcc)) * self.vbc[1:, :]

        # need to advance only interior points, therefore 1:-1
        v2dy = np.diff(0.5 * (v2l + v2r), axis=0)[:, 1:-1] / self.dy

        vul = (ucorner[:, 1:-1] + np.fabs(ucorner[:, 1:-1])) * self.vbc[1:-1, :-1]
        vur = (ucorner[:, 1:-1] - np.fabs(ucorner[:, 1:-1])) * self.vbc[1:-1, 1:]
        vudx = np.diff(0.5 * (vul + vur), axis=1) / self.dx

        # diffusion x
        dudx_cc = np.diff(self.ubc, axis=1) / self.dx
        dudy_corners = np.diff(self.ubc, axis=0) / self.dy

        # only for non-ghost points, hence 1:-1
        d2udx2 = np.diff(dudx_cc, axis=1)[1:-1, :] / self.dx

        # only for non-ghost points, hence 1:-1
        d2udy2 = np.diff(dudy_corners, axis=0)[:, 1:-1] / self.dy

        # diffusion y
        dvdy_cc = np.diff(self.vbc, axis=0) / self.dy
        dvdx_corners = np.diff(self.vbc, axis=1) / self.dx

        # only for non-ghost points, hence 1:-1
        d2vdy2 = np.diff(dvdy_cc, axis=0)[:, 1:-1] / self.dy

        # only for non-ghost points, hence 1:-1
        d2vdx2 = np.diff(dvdx_corners, axis=1)[1:-1, :] / self.dx

        self.ubc[1:-1, 1:-1] += self.dt * (
            self.visc * (d2udx2 + d2udy2) - (u2dx + uvdy)
        )
        self.vbc[1:-1, 1:-1] += self.dt * (
            self.visc * (d2vdx2 + d2vdy2) - (vudx + v2dy)
        )

    def project_velocity(self):
        """Project the velocity field."""
        self.ubc[1:-1, 1:-1] -= np.diff(self.p, axis=1)[1:-1, :] / self.dx
        self.vbc[1:-1, 1:-1] -= np.diff(self.p, axis=0)[:, 1:-1] / self.dy

    def apply_pres_bc(self):
        # West (homogenous neumann)
        self.p[:, 0] = self.p[:, 1]

        # East (Dirchlet)
        self.p[:, -1] = self.presbcvals[self.East]

        # South (homogenous neumann)
        self.p[0, :] = self.p[1, :]

        # North (homogenous neumann)
        self.p[-1, :] = self.p[-2, :]

    def solve_pressure_poisson(self):
        """Solve the Poisson equation for pressure."""
        self.apply_velocity_bc()
        # self.apply_pres_bc()
        velrhs = (
            np.diff(self.ubc, axis=1)[1:-1, 1:-1] / self.dx
            + np.diff(self.vbc, axis=0)[1:-1, 1:-1] / self.dy
        )
        rhs = self.presrhs + velrhs.flatten()
        psoln = np.linalg.solve(self.presmat, rhs)

        # update pressure
        self.p[1:-1, 1:-1] = psoln.reshape(self.ncells[1], self.ncells[0])
        self.apply_pres_bc()

    def computediv(self):
        dudx = np.diff(self.ubc[1:-1, 1:-1], axis=1) / self.dx
        dvdy = np.diff(self.vbc[1:-1, 1:-1], axis=0) / self.dy

        self.div[1:-1, 1:-1] = dudx + dvdy
        self.divl2norm = np.linalg.norm(self.div.flatten())


def yavg(f):
    """Calculate the forward average in y.
    :param f: data to average
    :type f: array
    :return: averaged data in x
    :rtype: array
    """
    return 0.5 * (f[1:, :] + f[:-1, :])


def xavg(f):
    """Calculate the forward average in x
    :param f: data to average
    :type f: array
    :return: averaged data in y
    :rtype: array
    """
    return 0.5 * (f[:, 1:] + f[:, :-1])
