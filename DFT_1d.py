import numpy as np
import matplotlib.pyplot as plt

from ks_solver import KS_Solver

import grid
# phisical constants
omega = 2*np.pi
m = 1
hbar = 1

# numeric paramters
FLOAT_PRECISION = 1e-05
Ngrid = 200
xmin = -5
xmax = 5

# molecule params:
NumberofElectrons = 3

# DFT params
maxiterations = 3
grid_1d = grid.Cartesian_1d(Ngrid,xmin, xmax) 
initial_density = NumberofElectrons* np.ones(Ngrid)/(Ngrid)

ks_1d = KS_Solver(grid_1d, NumberofElectrons, maxiterations, initial_density,convergence_threshold=0.001)

ks_1d.runSCF()
