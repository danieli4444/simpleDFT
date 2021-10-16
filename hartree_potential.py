import numpy as np
import scipy
from scipy.integrate.odepack import odeint
from physical_units import m_e, eps_0, e, hbar
from scipy import integrate
import matplotlib.pyplot as plt

from scipy import *
from scipy import integrate,interpolate

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

    sol = integrate.solve_ivp(ode_sys, t_span=[xvec[0], xvec[-1]] , y0=[0,1], t_eval=xvec, args=(dens, ),dense_output=True)

    U = sol.y[0]
    return U

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

    Ha_energy_constant = 0.5 
    # estimation at which r >> R the potential is similar to pointwise charge
    R = xvec[-3]
    # at r>>R for some large R, the hartree potential is similar to a pointwise charge 
    U_pointwise_potential = xvec*numelectrons/R 

    # DONT FORGET TO REMOVE THIS
    # xvec = xvec[::-1]
    # density_1s = np.exp(-2*xvec)/np.pi

    # U_poisson'' = - 4pi * r * density(r)
    U_poisson = solve_poisson_ode(xvec,density)
    #plt.plot(xvec,U_poisson,color='red')
    #plt.show()
    #print(U_poisson[-100]/xvec[-100])
    #print(U_pointwise_potential[-100]/xvec[-100])
    # normalization??
    #norm = integrate.simps(U_poisson**2,x=xvec)
    #U_poisson *= 1/np.sqrt(abs(norm))

    U_final = U_poisson +  U_pointwise_potential - (xvec/R) * U_poisson

    print("U_poisson = {0} , U_pointwise = {1}".format(U_poisson.max(), U_pointwise_potential.max()))
    Vhartree = U_final/xvec

    # get rid of the numeric artifacts
    for i in range(0,5):
        Vhartree[i] = 0
        Vhartree[-i] = 0

    # plt.plot(xvec, Vhartree, color='blue')
    # plt.plot(xvec,density,color='red')

    for t in range(len(Vhartree)):
        r = xvec[t]
        d = density[t] 
        Ehartree = 0
        Ehartree += Vhartree[t] * d * r**2 * grid_dr
    
    Ehartree *= Ha_energy_constant 
    Ehartree2 = integrate.simps(Vhartree* density * xvec**2 ,xvec)
    #Ehartree = Ha_energy_constant * 4*np.pi (Vhartree*density*xvec**2,xvec)
    print("solved Vhartree using poisson!")
    print("Ehartree = {0} ".format(Ehartree))
    print("Ehartree with scipy.simps integration = {0}".format(Ehartree2))

    return float(Ehartree2) , np.asarray(np.mat(np.diag(Vhartree, 0)))




if __name__ == "__main__":
    """this code purpose is to check the ode poisson solution on a simple 
    gaussian function which integral is known analyticly
    """

    Ngrid = 2000
    rmax = 40.0
    rmin = 1e-10
    (xvec, grid_dr) = np.linspace(rmin, rmax, Ngrid,retstep=True)
    numelectrons=1
    R=rmax

    #true_computed_density = check_poisson(xvec,grid_dr)
    # g = gaussian_density(xvec)
    # x = -4*np.pi *xvec*g
    # plt.plot(xvec,x,color='blue',label="ground truth-4pi*x *p(x)")
    # U_potential = solve_poisson_ode(xvec,g)
    # computed_density = check_potential(U_potential,xvec,grid_dr)
    # plt.plot(xvec[3:-5],computed_density[3:-5],color='red',label="solved d2U/d2r")
        
    density_1s = np.exp(-2*xvec)/np.pi
    U_potential_1s = solve_poisson_ode(xvec,density_1s)
    U_pointwise_potential = xvec*numelectrons/rmax
    U_final = U_potential_1s +  U_pointwise_potential - (xvec/rmax) * U_potential_1s
    V_potential_1s = U_final/xvec
    xvec=xvec
    density_1s = density_1s
    plt.plot(xvec,U_potential_1s,color='blue')
    plt.show()
    E_1s = 4*np.pi * integrate.simps(xvec**2 * density_1s * V_potential_1s,xvec)
    print("the Hartree energy for 1s Hydrodgen is (should be 0.625):",E_1s)
    

