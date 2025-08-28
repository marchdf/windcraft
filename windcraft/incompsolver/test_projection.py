from solver import Solver
from plotcfd import plotflowfields
import numpy as np

# main
n = 40
xmax = 1.0
ymax = 1.0
solverobj = Solver(n, xmax, ymax)
dx = xmax / (n - 1)
dy = ymax / (n - 1)

# test projection, set sinusoidal values for u
x = np.linspace(0.0, xmax, num=n)
y = np.linspace(dy / 2, ymax - dy / 2, num=n - 1)

for j in range(n - 1):
    for i in range(n):
        solverobj.ubc[j + 1][i + 1] = np.sin(2 * np.pi * x[i] * 5) * np.sin(
            2 * np.pi * y[j] * 5
        )

for j in range(n):
    for i in range(n - 1):
        solverobj.vbc[j + 1][i + 1] = np.sin(2 * np.pi * y[i] * 10) * np.sin(
            2 * np.pi * x[j] * 10
        )

solverobj.apply_velocity_bc()
solverobj.computediv()
print("div before:", solverobj.divl2norm)
solverobj.solve_pressure_poisson()
solverobj.project_velocity()
solverobj.computediv()
print("div after:", solverobj.divl2norm)

# solverobj.solve(2)
plotflowfields(
    solverobj.ubc,
    solverobj.vbc,
    solverobj.p,
    solverobj.div,
    0.0,
    xmax,
    0.0,
    ymax,
    solverobj.ncells[0],
    solverobj.ncells[1],
    1,
    1.0,
    1.0,
    1.0,
)
