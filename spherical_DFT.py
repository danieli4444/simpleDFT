import numpy as np
import matplotlib.pyplot as plt

from ks_solver import KS_Solver

import grid
# phisical constants
#omega = 2*np.pi
#m = 1
#hbar = 1

# numeric paramters
FLOAT_PRECISION = 1e-05
Ngrid = 500
rmin = 2e-10
rmax = 0.0

# molecule params:
NumberofElectrons = 2

# DFT params
maxiterations = 3
grid_1d = grid.Radial_Grid(Ngrid,rmin, rmax) 
initial_density = NumberofElectrons* np.ones(Ngrid)/(Ngrid)

ks_1d = KS_Solver(grid_1d, NumberofElectrons, maxiterations, initial_density,convergence_threshold=0.001)

ks_1d.runSCF()
