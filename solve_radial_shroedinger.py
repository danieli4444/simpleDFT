""" 
create simple Hydrogen Radial Hamiltonian and find ground state energy
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.linalg import eigvals
from scipy import constants as const

# phisical constants
omega = 2*np.pi
m_e = const.m_e
hbar = const.hbar
e = const.e
eps_0 = const.epsilon_0

# grid constants:
R_gridsize = 2000
Rmin = 2e-9
Rmax = 0.0

"""
phy_gridsize = 200
theta_gridsize = 200
phy_min= 0
phy_max = np.pi * 2
theta_min = 0
theta_max = np.pi
"""


# the first term is just double differentiation:
def get_kinetic_mat(r_gridsize, rmin, rmax):
    """ assuming u = r * R sustitution

    Args:
        r_gridsize ([type]): [description]
        rmin ([type]): [description]
        rmax ([type]): [description]

    Returns:
        [type]: [description]
    """
    (rvec, dr) = np.linspace(rmin, rmax, r_gridsize, retstep=True)
    dia = -2*np.ones(r_gridsize)
    offdia = np.ones(r_gridsize-1)
    d2grid = np.mat(np.diag(dia, 0) + np.diag(offdia, -1) + \
                    np.diag(offdia, 1))/dr**2
    # avoid strange things at the edge of the grid
    #d2grid[0, :] = 0
    #d2grid[Ngrid-1, :] = 0
    d2grid[-1,-1] = d2grid[0,0]
    Ekin = -hbar**2/(2*m_e) * d2grid
    return np.asarray(Ekin)

# potential energy:


def get_potential_term1(r_gridsize, rmin, rmax):
    """ assuming u = r * R sustitution: e**2/4pieps_0 * 1/r

    Args:
        r_gridsize ([type]): [description]
        rmin ([type]): [description]
        rmax ([type]): [description]

    Returns:
        [type]: [description]
    """
    (rvec, dr) = np.linspace(rmin, rmax, r_gridsize, retstep=True)
    Vr = -(e**2/(4*np.pi * eps_0)) * 1/rvec  # simple 1/r potential
    Vr = np.diag(Vr, 0)
    return np.asarray(Vr)

def get_potential_term2(r_gridsize, rmin, rmax):
    """ assuming u = r * R sustitution: [l(l+1)*h_bar**2/2*m] * 1/r**2

    Args:
        r_gridsize ([type]): [description]
        rmin ([type]): [description]
        rmax ([type]): [description]

    Returns:
        [type]: [description]
    """
    (rvec, dr) = np.linspace(rmin, rmax, r_gridsize, retstep=True)
    Vr = (hbar**2 /2*m_e) * 1/rvec**2  # simple 1/r potential
    Vr = np.diag(Vr, 0)
    return np.asarray(Vr)

def diagonalize_hamiltonian(H):
    """ solves Kohn Sham Equations (hamiltonian diagonalization)
        returns sorted EigenVals,EigenVecs
    Args:
        H ([ndarray (2d)]): hamiltonian to diagonalize
    """
    print("Hamiltonian Diagonaliztion...")
    epsilon_n, psi_gn = np.linalg.eigh(H)
    sortinds = np.argsort(epsilon_n)
    sorted_EigVecs = psi_gn[:, sortinds]
    sorted_EigVals = epsilon_n[sortinds]
    return sorted_EigVals, sorted_EigVecs

def plot_densities(r, densities, eigenvalues):
    plt.xlabel('x ($\\mathrm{\AA}$)')
    plt.ylabel('probability density ($\\mathrm{\AA}^{-1}$)')
     
    energies = ['E = {: >5.2f} eV'.format(eigenvalues[i] / e) for i in range(3)]
    plt.plot(r * 1e+10, densities[0], color='blue',  label=energies[0])
    plt.plot(r * 1e+10, densities[1], color='green', label=energies[1])
    plt.plot(r * 1e+10, densities[2], color='red',   label=energies[2])
     
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    (Rvec, d_r) = np.linspace(Rmin, Rmax, R_gridsize,retstep=True,endpoint=False)
    #(phy_vec, d_phy) = np.linspace(phy_min, phy_max, phy_gridsize,retstep=True)
    #(theta_vec, d_theta)= np.linspace(theta_min, theta_max, theta_gridsize,retstep=True)
    #theta_vec, phy_vec = np.meshgrid(theta_vec, phy_vec)

    kinetic_term = get_kinetic_mat(R_gridsize,Rmin,Rmax)
    potential_term1 = get_potential_term1(R_gridsize,Rmin,Rmax)
    potential_term2 = get_potential_term2(R_gridsize,Rmin,Rmax)

    H = kinetic_term + potential_term1 + potential_term2
    (Eigen_Vals,Eigen_Vecs) = diagonalize_hamiltonian(H)

    """ compute probability density for each eigenvector """
    densities = [np.absolute(Eigen_Vecs[i, :])**2 for i in range(len(Eigen_Vals))]

    #plot_densities(Rvec, densities, Eigen_Vals)
    plt.plot(Rvec*1e10,Eigen_Vecs[0]**2)
    plt.show()

# visualization:

# R = 1
# X = R * np.sin(phy_vec) * np.cos(theta_vec)
# Y = R * np.sin(phy_vec) * np.sin(theta_vec)
# Z = R * np.cos(phy_vec)



# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# plot = ax.plot_surface(
#     X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
#     linewidth=0, antialiased=False, alpha=0.5)

# plt.show()
