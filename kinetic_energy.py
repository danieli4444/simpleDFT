import numpy as np
from physical_units import m_e, eps_0, e, hbar

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
    Ekin = hbar**2/(2*m_e)*d2grid
    return np.asarray(Ekin)