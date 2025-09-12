Numerical methods
=================

The CFD solver solves the incompressible Euler equations on a
staggered grid:

.. math::

   \frac{\partial u}{\partial t} + \frac{\partial p}{\partial x} = - \frac{\partial (u^2)}{\partial x} - \frac{\partial (u v)}{\partial y}

   \frac{\partial v}{\partial t} + \frac{\partial p}{\partial y} = - \frac{\partial (uv)}{\partial x} - \frac{\partial (v^2)}{\partial y}

   \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0


The advective part is solved explicitly using upwinding (smooth
transition between central differencing and upwinding). The pressure
Poisson equation is solved using iteration based methods. The turbines
in the wind farm are represented by an additional source term which
uses an actuator disk model to model the velocity deficit. Inspiration
for the CFD solver come from `sAlexander's Fortran solver
<https://github.com/sAlexander/cfd>`_ and follows a description found
`here <http://www-math.mit.edu/cse/codes/mit18086_navierstokes.pdf>`_.


