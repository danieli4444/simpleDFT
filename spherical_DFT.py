import numpy as np
import matplotlib.pyplot as plt

from ks_solver import KS_Solver

import grid


# numeric paramters
FLOAT_PRECISION = 1e-05
Ngrid = 1000
rmin = 1e-09
rmax = 1e-15

# molecule params:
NumberofElectrons = 1

# DFT params
maxiterations = 3
grid_1d = grid.Radial_Grid(Ngrid,rmin, rmax) 
initial_density = NumberofElectrons* np.ones(Ngrid)/(Ngrid)

ks_1d = KS_Solver(grid_1d, NumberofElectrons, maxiterations, initial_density,convergence_threshold=0.001)

ks_1d.runSCF()
