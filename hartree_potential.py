import numpy as np
from scipy import constants as const

# phisical constants
omega = 2*np.pi
m_e = const.m_e 
hbar = const.hbar
e = const.e
eps_0 = const.epsilon_0

def calculate_hartree_pot(density,xvec,grid_dr, eps=1e-12):
    Ha_energy = 0
    Ha_potential = np.zeros(xvec.shape)
    Ha_energy_constant = - 0.5 * e**2
    Ha_potential_constant = - e
    for t1 in range(density.size):
        x1 = xvec[t1]
        d1 = density[t1]
        for t2 in range(density.size):
            x2 = xvec[t2]
            d2 = density[t2]
            Ha_energy = Ha_energy + grid_dr**2*(d1*d2)/np.sqrt(np.power(x1-x2,2) + eps)
            Ha_potential[t1] = Ha_potential[t1] + grid_dr*d1/np.sqrt(np.power(x1-x2,2) + eps)

    Ha_energy *= Ha_energy_constant
    Ha_potential = Ha_potential_constant * np.mat(np.diag(Ha_potential, 0))
    return float(Ha_energy), np.asarray(Ha_potential)