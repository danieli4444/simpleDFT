from scipy import constants as const
import numpy as np

units = "AU"

# config grid dimensions
if units == "AU":
    rmin = 1e-05
    rmax = 40
else:
    rmin = 1e-09
    rmax = 1e-15

# config physical constants
if units == "AU":
    m_e = 1
    e = 1
    eps_0 = 1/(4*np.pi)
    hbar = 1

else:
    # phisical constants
    omega = 2*np.pi
    m_e = const.m_e 
    hbar = const.hbar
    e = const.e
    eps_0 = const.epsilon_0
