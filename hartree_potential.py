import numpy as np
from scipy.integrate.odepack import odeint
from physical_units import m_e, eps_0, e, hbar
from scipy import integrate
from numerov import runNumerov
import matplotlib.pyplot as plt

from scipy import *
from scipy import integrate,interpolate
from scipy.integrate import solve_ivp,odeint

import math
from kinetic_energy import get_kinetic_mat

def gaussian_density(xvec):
    sigma = 1
    g_const = sigma**3 * np.power((2*np.pi),1.5)
    g_const = 1/g_const
    a = -xvec**2 /(2* sigma**2)
    g = g_const * np.exp(a)
    return g

def gaussian_potential(xvec,grid_dr):
    sigma = 1
    g_v = np.zeros(len(xvec))
    for i in range(0,len(xvec)):
        a = math.erf(xvec[i]/(np.sqrt(2) * sigma))
        g_v[i] = a
    return g_v

def check_potential(potential,xvec,grid_dr):
    d2 = - 2* get_kinetic_mat(xvec,grid_dr)
    computed_density = np.dot(d2,potential) 
    return computed_density

def check_poisson(xvec,grid_dr):
    
    g_pot = gaussian_potential(xvec,grid_dr)
    d2 = - 2* get_kinetic_mat(xvec,grid_dr)
    computed_density = np.dot(d2,g_pot) 
    return computed_density


def calculate_hartree_pot(density, xvec, grid_dr, numelectrons, eps=1e-12):
    """ U_final = U_poisson +  U_pointwise_potential - (xvec/R) * U_poisson
        where -  Vhartree = U_final / r
        (U /r is a trial solution for Vhartree)

        we calculate U_poisson using Numerov with boundry conditions U(0)=0 , U'(0)=1

        more details at - NUMERICAL SOLUTION OF KOHN–SHAM EQUATION FOR ATOM , Zbigniew Romanowski
        https://www.actaphys.uj.edu.pl/fulltext?series=Reg&vol=38&page=3263

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

    Ha_potential = np.zeros(xvec.shape)
    Ha_energy_constant = 0.5 
    Ha_potential_constant = 4*np.pi/eps_0 *e**2
    # estimation at which r >> R the potential is similar to pointwise charge
    grid_center = int(len(xvec)/2)
    R = xvec[grid_center]
    # at r>>R for some large R, the hartree potential is similar to a pointwise charge 
    U_pointwise_potential = xvec*numelectrons/R 

    # U_poisson'' = - 4pi * r * density
    g = -4*np.pi * xvec * density
    U_0 = 0.0001
    U_diff_0 = 1.000001
    U_poisson = runNumerov(g,U_0,U_diff_0,grid_dr)[::-1]
    print(U_poisson[-100])
    print(U_pointwise_potential[-100])
    # normalization??
    #norm = integrate.simps(U_poisson**2,x=xvec)
    #U_poisson *= 1/np.sqrt(abs(norm))

    U_final = U_poisson +  U_pointwise_potential - (xvec/R) * U_poisson

    print("U_poisson = {0} , U_pointwise = {1}".format(U_poisson.max(), U_pointwise_potential.max()))
    Vhartree = U_final/xvec
    plt.plot(xvec, Vhartree, color='blue')
    plt.plot(xvec,density,color='red')
    plt.plot(xvec,U_pointwise_potential/xvec,color='black')
    plt.show()
    for t in range(density.size):
        r = xvec[t]
        d = density[t] 
        Ehartree = 0
        Ehartree += Vhartree[t] * d * r**2 * grid_dr
    
    Ehartree *= Ha_energy_constant * 4*np.pi
    
    #Ehartree = Ha_energy_constant * 4*np.pi (Vhartree*density*xvec**2,xvec)
    print("solved Vhartree using poisson!")
    print("Ehartree = {0} ".format(Ehartree))

    return float(Ehartree) , np.asarray(np.mat(np.diag(Vhartree, 0)))
    # for t1 in range(density.size):
    #     r1 = xvec[t1]
    #     d1 = density[t1]
    #     for t2 in range(density.size):
    #         r2 = xvec[t2]
    #         d2 = density[t2]
    #         if r1 < r2:
    #             Ha_potential[t1] += (1/r1) * grid_dr * r2**2 * d2 
    #         else:
    #             Ha_potential[t1] += (1/r1) * grid_dr * r2 * d2 
        
    #     Ha_energy += Ha_energy_constant *Ha_potential_constant *Ha_potential[t1]*d1*r1**2*grid_dr

    # Ha_potential = Ha_potential_constant * np.mat(np.diag(Ha_potential, 0))
    return float(Ha_energy), np.asarray(Ha_potential)

def solve_poisson_ode(xvec,density):
    """ The poisson Eq is: 1/r * d2/dr2 (r *V) = -4 pi * density(r)
        which can be transformed to -
        d2/dr2 (U) = -4pi*density(r) * r ; where V = U/r

        The boundry conditions are: U(0) = 0, U(r>>R)=Num_electrons (pointwise charge potential)

        in order to solve, we transform into 2 first order equations:
        (1) U'(r) = y(r)
        (2) y'(r) = -4pi * density * r
    
    """
        
    def ode_sys(t, Y,dens):

        y = Y[0]
        dx_dt=Y[1]
        d2x_dt2= -4*np.pi *t*  dens(t)
        return [dx_dt, d2x_dt2]

    dens = interpolate.interp1d(x=xvec, y=density)

    sol = solve_ivp(ode_sys, t_span=[0, 10] , y0=[0,1], t_eval=xvec, args=(dens, ),dense_output=True)

    return sol.y[0]
    




if __name__ == "__main__":
    """this code purpose is to check the ode poisson solution on a simple 
    gaussian function which integral is known analyticly
    """

    Ngrid = 1000
    rmin = 0.0
    rmax = 10.0
    (xvec, grid_dr) = np.linspace(rmin, rmax, Ngrid,retstep=True)
    #true_computed_density = check_poisson(xvec,grid_dr)
    
    g = gaussian_density(xvec)
    x = -4*np.pi *xvec*g
    plt.plot(xvec,x,color='blue',label="ground truth-4pi*x *p(x)")

    sol = solve_poisson_ode(xvec,g)
    computed_density = check_potential(sol,xvec,grid_dr)
    plt.plot(xvec[3:-5],computed_density[3:-5],color='red',label="solved d2U/d2r")
    
    plt.legend()
    plt.show()

