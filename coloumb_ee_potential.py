import numpy as np

def calculate_coloumb(density,xvec,grid_dr, eps=1e-1):
    c_energy = 0
    c_potential = np.zeros(xvec.shape)
    for t1 in range(density.size):
        x1 = xvec[t1]
        d1 = density[t1]
        for t2 in range(density.size):
            x2 = xvec[t2]
            d2 = density[t2]
            c_energy = c_energy + 0.5*grid_dr**2*(d1*d2)/np.sqrt(np.power(x1-x2,2) + eps)
            c_potential[t1] = c_potential[t1] + grid_dr*d1/np.sqrt(np.power(x1-x2,2) + eps)

    c_potential = np.mat(np.diag(c_potential, 0))

    return float(c_energy), np.asarray(c_potential)