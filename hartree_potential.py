import numpy as np
from physical_units import m_e, eps_0, e, hbar

def calculate_hartree_pot(density, xvec, grid_dr, numelectrons, eps=1e-12):
    """ U_final = U_poisson +  U_pointwise_potential - (xvec/R) * U_poisson
        where -  Vhartree = U_final / r
        (U /r is a trial solution for Vhartree)

    Args:
        density ([type]): [description]
        xvec ([type]): [description]
        grid_dr ([type]): [description]
        numelectrons ([type]): [description]
        eps ([type], optional): [description]. Defaults to 1e-12.

    Returns:
        [type]: [description]
        [type]: [description]
    """
    
    Ha_energy = 0
    Ha_potential = np.zeros(xvec.shape)
    Ha_energy_constant = - 0.5 
    Ha_potential_constant = 4*np.pi *e**2
    # estimation at which r >> R the potential is similar to pointwise charge
    R = xvec.max() * 10
    # at r>>R for some large R, the hartree potential is similar to a pointwise charge 
    U_pointwise_potential = xvec*numelectrons/R 


    

    for t1 in range(density.size):
        r1 = xvec[t1]
        d1 = density[t1]
        for t2 in range(density.size):
            r2 = xvec[t2]
            d2 = density[t2]
            if r1 < r2:
                Ha_potential[t1] += (1/r1) * grid_dr * r2**2 * d2 
            else:
                Ha_potential[t1] += (1/r1) * grid_dr * r2 * d2 
        
        Ha_energy += Ha_energy_constant *Ha_potential_constant *Ha_potential[t1]*d1*r1**2*grid_dr

    Ha_potential = Ha_potential_constant * np.mat(np.diag(Ha_potential, 0))
    return float(Ha_energy), np.asarray(Ha_potential)