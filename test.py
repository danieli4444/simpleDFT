import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace

Ngrid = 200
xmin = -5
xmax = 5

#(xvec, dX) = np.linspace(xmin, xmax, Ngrid,retstep=True)
#print("dX = {0}".format(dX))
(xvec,dX) = np.linspace(xmin, xmax, Ngrid,retstep=True)
print(dX)
y = np.sin(xvec)
plt.plot(xvec,y,label="sin(x)")
# laplacian matrix
dia = np.ones(Ngrid)
offdia = np.ones(Ngrid-1)
d2 = np.zeros((Ngrid,Ngrid))
d2 = d2 + -2*np.diag(dia) + np.diag(offdia,k=1) + np.diag(offdia,k=-1)
d2 = 0.5 * d2/(dX**2)
z = np.dot(d2,y)
plt.plot(xvec[1:-1],z[1:-1], label="d2(-sinx)")

plt.show()



