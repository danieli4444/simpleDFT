import numpy as np
from scipy import constants as const

# phisical constants
omega = 2*np.pi
m_e = const.m_e 
hbar = const.hbar
e = const.e
eps_0 = const.epsilon_0

# m_e = 1
# e = 1
# eps_0 = 1
# hbar = 1

def get_kinetic_mat(xvec, grid_dr):
    dx = grid_dr
    dia = -2*np.ones(xvec.size)
    offdia = np.ones(xvec.size-1)
    d2grid = np.mat(np.diag(dia, 0) + np.diag(offdia, -1) + \
                    np.diag(offdia, 1))/dx**2
    # avoid strange things at the edge of the grid
    #d2grid[0, :] = 0
    #d2grid[Ngrid-1, :] = 0
    d2grid[-1,-1] = d2grid[0,0]
    Ekin = -hbar**2/(2*m_e)*d2grid
    return np.asarray(Ekin)