import numpy as np
import matplotlib.pyplot as plt
import time

from numpy.core.function_base import linspace
from numpy.linalg import eigvals
import coloumb_ee_potential
import external_potential
import xc_potential



# numeric paramters
FLOAT_PRECISION = 1e-05
Ngrid = 200
xmin = -5
xmax = 5
(xvec, dX) = np.linspace(xmin, xmax, Ngrid,retstep=True)
print("dX = {0}".format(dX))

