from solver import Solver
from plotcfd import plotflowfields

# main
n = 16
xmax = 10.0
ymax = 1.0
solverobj = Solver(n, xmax, ymax)

solverobj.solve(30)
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
