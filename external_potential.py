import numpy as np
from physical_units import m_e, eps_0, e, hbar

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

