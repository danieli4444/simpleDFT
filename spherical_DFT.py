import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from ks_solver import KS_Solver
import grid
from physical_units import units,rmin,rmax

# numeric paramters
FLOAT_PRECISION = 1e-02
Ngrid = 3000

# remember set units choice in the physical_units.py file
print("using {0} units".format(units))

# molecule params:
NumberofElectrons = 1
# DFT params
maxiterations = 30
grid_1d = grid.Radial_Grid(Ngrid, rmin, rmax) 
#initial_density = np.exp(-2*grid_1d.gridvec)/np.pi
initial_density = NumberofElectrons**4 /(64*np.pi) * np.exp(-NumberofElectrons*grid_1d.gridvec/2)
#initial_density =   np.exp(-2*grid_1d.gridvec)/np.pi

#check_density = 4*np.pi *integrate.simps(initial_density*grid_1d.gridvec**2,grid_1d.gridvec)
#print(check_density)
ks_1d = KS_Solver(grid_1d, NumberofElectrons, maxiterations, initial_density,convergence_threshold=0.001)

ks_1d.runSCF()
