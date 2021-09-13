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

def get_external_harmonic_potential(xvec):
    """simple x^2 for testing

    Args:
        xvec ([type]): [description]

    Returns:
        [type]: [description]
    """
    Vx = xvec**2  # simple X^2 potential
    Vx = np.diag(Vx, 0)
    return np.asarray(Vx)


def get_radial_potential_term(rvec,numelectrons):
    """ assuming u = r * R sustitution: e**2/4pieps_0 * 1/r

    Args:
        r_gridsize ([type]): [description]
        rmin ([type]): [description]
        rmax ([type]): [description]

    Returns:
        [type]: [description]
    """
    Vr = -1 * (numelectrons*e**2/(4*np.pi * eps_0)) * 1/rvec  # simple 1/r potential
    Vr = np.diag(Vr, 0)
    return np.asarray(Vr)