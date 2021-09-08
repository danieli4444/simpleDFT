""" 
create simple Hydrogen Radial Hamiltonian and find probability densities and energies
without any angular elemnts (l=0).
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
Rmin = 0.0
Rmax = 2e-9

"""
phy_gridsize = 200
theta_gridsize = 200
phy_min= 0
phy_max = np.pi * 2
theta_min = 0
theta_max = np.pi
"""


# the first term is just double differentiation:
def calc_laplace_term(rvec):
    """ assuming u = r * R sustitution

    Args:
        r_gridsize ([type]): [description]
        rmin ([type]): [description]
        rmax ([type]): [description]

    Returns:
        [type]: [description]
    """
    dr = rvec[1] - rvec[0]
    dia = -2*np.ones(rvec.size)
    offdia = np.ones(rvec.size-1)
    d2grid = np.mat(np.diag(dia, 0) + np.diag(offdia, -1) + \
                    np.diag(offdia, 1))/dr**2
    # avoid strange things at the edge of the grid
    #d2grid[0, :] = 0
    #d2grid[Ngrid-1, :] = 0
    d2grid[-1,-1] = d2grid[0,0]
    Ekin = hbar**2/(2*m_e) * d2grid
    return np.asarray(Ekin)

# potential energy:


def get_potential_term(rvec):
    """ assuming u = r * R sustitution: e**2/4pieps_0 * 1/r

    Args:
        r_gridsize ([type]): [description]
        rmin ([type]): [description]
        rmax ([type]): [description]

    Returns:
        [type]: [description]
    """
    Vr = (e**2/(4*np.pi * eps_0)) * 1/rvec  # simple 1/r potential
    Vr = np.diag(Vr, 0)
    return np.asarray(Vr)

def get_angular_term(rvec,l_number):
    """ assuming u = r * R sustitution: [l(l+1)*h_bar**2/2*m] * 1/r**2

    Args:
        r_gridsize ([type]): [description]
        rmin ([type]): [description]
        rmax ([type]): [description]

    Returns:
        [type]: [description]
    """
    Vr = (hbar**2 /2*m_e) * l_number*(l_number+1)/rvec**2  # simple 1/r potential
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
    plt.title('Hydrogen densities for l=0, n=1,2,3') 
    energies = ['E = {: >5.2f} eV'.format(eigenvalues[i] / e) for i in range(3)]
    plt.plot(r * 1e+10, densities[0], color='blue',  label=energies[0])
    plt.plot(r * 1e+10, densities[1], color='green', label=energies[1])
    plt.plot(r * 1e+10, densities[2], color='red',   label=energies[2])
     
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    Rvec = np.linspace(2e-9, 0.0, R_gridsize,endpoint=False)
    #(phy_vec, d_phy) = np.linspace(phy_min, phy_max, phy_gridsize,retstep=True)
    #(theta_vec, d_theta)= np.linspace(theta_min, theta_max, theta_gridsize,retstep=True)
    #theta_vec, phy_vec = np.meshgrid(theta_vec, phy_vec)

    kinetic_term = calc_laplace_term(Rvec)
    potential_term = get_potential_term(Rvec)
    print(potential_term)
    l_number = 0
    #angular_term = get_angular_term(Rvec, l_number)
    angular_term =0
    # hamiltonian = -hbar**2 / (2.0 * m_e) * (laplace_term - angular_term) - potential_term

    H = -kinetic_term + angular_term - potential_term 
    (Eigen_Vals,Eigen_Vecs) = diagonalize_hamiltonian(H)

    """ compute probability density for each eigenvector """
    densities = [np.absolute(Eigen_Vecs[:, i])**2 for i in range(len(Eigen_Vals))]

    plot_densities(Rvec, densities, Eigen_Vals)
    #plt.plot(Rvec*1e10,Eigen_Vecs[0]**2)
    #plt.show()

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
