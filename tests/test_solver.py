# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

import unittest
import numpy as np
import numpy.testing as npt
import windcraft as wc


class SolverTestCase(unittest.TestCase):
    """Tests for `solver.py`."""

    def setUp(self):
        self.solver = wc.Solver(400, 300)

        # Random initial perturbation
        self.x, _ = np.meshgrid(
            np.linspace(
                0 + self.solver.dx,
                self.solver.width - self.solver.dx,
                self.solver.n[0] - 1,
            ),
            np.linspace(
                0 + 0.5 * self.solver.dx,
                self.solver.height - 0.5 * self.solver.dx,
                self.solver.n[1],
            ),
            indexing="ij",
        )
        _, self.y = np.meshgrid(
            np.linspace(
                0 + 0.5 * self.solver.dx,
                self.solver.width - 0.5 * self.solver.dx,
                self.solver.n[0],
            ),
            np.linspace(
                0 + self.solver.dx,
                self.solver.height - self.solver.dx,
                self.solver.n[1] - 1,
            ),
            indexing="ij",
        )

        self.solver.u = np.cos(2 * np.pi * self.x)
        self.solver.v = np.sin(2 * np.pi * self.y)
        np.random.seed(42)
        self.solver.p = 0.001 * np.random.rand(self.solver.n[0], self.solver.n[1])

    def test_reset(self):
        """Is the reset function correct?"""
        self.solver.reset()
        npt.assert_almost_equal(self.solver.u, self.solver.u0)
        npt.assert_almost_equal(self.solver.v, 0.0)
        npt.assert_almost_equal(self.solver.p, 0.0)

    def test_apply_velocity_bc(self):
        """Is the velocity bc function correct?"""
        self.solver.apply_velocity_bc()
        npt.assert_almost_equal(np.linalg.norm(self.solver.ubc), 89.37910769214193)
        npt.assert_almost_equal(np.linalg.norm(self.solver.vbc), 81.86848953639407)

    def test_advection(self):
        """Is the advection function correct?"""
        self.solver.apply_velocity_bc()
        self.solver.advection()

        npt.assert_almost_equal(np.linalg.norm(self.solver.u), 79.35577232785283)
        npt.assert_almost_equal(np.linalg.norm(self.solver.v), 81.62214012672611)

    def test_project_velocity(self):
        """Is the velocity projection function correct?"""
        self.solver.project_velocity()
        npt.assert_almost_equal(np.linalg.norm(self.solver.u), 79.15001074667737)
        npt.assert_almost_equal(np.linalg.norm(self.solver.v), 81.68592292141315)


if __name__ == "__main__":
    unittest.main()
