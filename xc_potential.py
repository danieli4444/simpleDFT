import numpy as np
from physical_units import m_e, eps_0, e, hbar
from scipy import integrate

#ToDo: check if dX is needed in the energy calc here??
def calculate_exchange(density,xvec,grid_dr):
    x_energy_const = -(3/4)*np.power((3/np.pi),1/3)
    x_potential_const = np.power((3/np.pi),1/3)
    
    #x_energy = x_energy_const * np.sum(np.power(density,4/3)) * grid_dr
    x_energy = 4 * np.pi *integrate.simps(np.power(density,4/3)* density * xvec**2)

    x_potential = x_potential_const * np.power(density,1/3)
    
    x_potential = np.mat(np.diag(x_potential, 0))


    return x_energy, np.asarray(x_potential)
