import mfem.ser as mfem
from .solver.solvers import AdvectionSolver
import numpy as np



@mfem.jit.vector(vdim=4, interface="c++")
def InitCond(x, out):
    # "Fast vortex"
    radius = 0.2
    Minf = 0.5
    beta = 1. / 5.

    xc = 0.0
    yc = 0.0
    # Nice units
    vel_inf = 1.
    den_inf = 1.

    specific_heat_ratio = 1.4
    gas_constant = 1.0

    pres_inf = (den_inf / specific_heat_ratio) * \
        (vel_inf / Minf) * (vel_inf / Minf)
    temp_inf = pres_inf / (den_inf * gas_constant)

    r2rad = 0.0
    r2rad += (x[0] - xc) * (x[0] - xc)
    r2rad += (x[1] - yc) * (x[1] - yc)
    r2rad /= (radius * radius)

    shrinv1 = 1.0 / (specific_heat_ratio - 1.)

    velX = vel_inf * \
        (1 - beta * (x[1] - yc) / radius * np.exp(-0.5 * r2rad))
    velY = vel_inf * beta * (x[0] - xc) / radius * np.exp(-0.5 * r2rad)
    vel2 = velX * velX + velY * velY

    specific_heat = gas_constant * specific_heat_ratio * shrinv1

    temp = temp_inf - (0.5 * (vel_inf * beta) *
                       (vel_inf * beta) / specific_heat * np.exp(-r2rad))

    den = den_inf * (temp/temp_inf)**shrinv1
    pres = den * gas_constant * temp
    energy = shrinv1 * pres / den + 0.5 * vel2

    out[0] = den
    out[1] = den*velX
    out[2] = den*velY
    out[3] = den*energy

mesh = mfem.Mesh("./mesh/periodic-segment.mesh")
EulerSolver()